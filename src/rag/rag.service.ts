import {
  BadRequestException,
  Injectable,
  InternalServerErrorException,
  ServiceUnavailableException,
} from '@nestjs/common';

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from '@langchain/google-genai';

import { Chroma } from '@langchain/community/vectorstores/chroma';
import { Document } from '@langchain/core/documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChromaClient } from 'chromadb';

import { AskRagDto } from './dto/ask-rag.dto';

type QueryRoute = 'PROFILE_ONLY' | 'INTERACTIONS_ONLY' | 'BOTH';
type LlmRagResponse = {
  answer: string;
  updatedSummary: string;
};

type KnowledgeCollection = 'DRUG_PROFILES' | 'DRUG_INTERACTIONS';

type SearchResult = [document: Document, distance: number];

type RoutedSearchResult = {
  collection: KnowledgeCollection;
  document: Document;
  distance: number;
};

type RagResources = {
  profileStore: Chroma;
  interactionStore: Chroma;
  model: ChatGoogleGenerativeAI;
};

type RetrievedSource = {
  rank: number;
  distance: number;
  collection: KnowledgeCollection;
  source: string;
  documentKind: string;
  content: string;

  profile: null | {
    jsonIndex: number | null;
    tradeName: string;
    genericName: string;
    activeIngredients: string;
    therapeuticCategory: string;
    diseaseCategory: string;
    dosageForm: string;
    packSize: string;
    pageRef: string;
  };

  interaction: null | {
    entryIndex: number | null;
    interactionIndex: number | null;
    subjectName: string;
    subjectType: string;
    relatedName: string;
    relatedType: string;
    appliesToScope: string;
    appliesToMembers: string;
    causingEntity: string;
    affectedEntity: string;
    effectType: string;
    severity: string;
    evidence: string;
    actionCategory: string;
    printedPage: string;
    pdfPage: string;
    sourceReference: string;
  };
};

@Injectable()
export class RagService {
  /**
   * Both collections live inside the same physical ChromaDB directory,
   * exposed by the same Chroma HTTP server.
   */
  private readonly chromaUrl =
    process.env.EXTRACTED_DRUGS_CHROMA_URL ?? 'http://localhost:8001';

  /** Drug information / profile collection. */
  private readonly profileCollectionName =
    process.env.EXTRACTED_DRUGS_COLLECTION ?? 'extracted_drug_rows';

  /** Drug-drug interaction collection. */
  private readonly interactionCollectionName =
    process.env.DRUG_INTERACTIONS_COLLECTION ?? 'drug_interactions';

  /**
   * Query embeddings MUST use the same model that was used while
   * creating both Chroma collections.
   */
  private readonly embeddingModel =
    process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-001';

  private readonly chatModel =
    process.env.GEMINI_CHAT_MODEL ?? 'gemini-3.1-flash-lite';

  private readonly defaultTopK = Number(process.env.RAG_TOP_K ?? '4');

  private resourcesPromise?: Promise<RagResources>;

  /**
   * Verify that both collections are reachable.
   * This method does not call the Gemini chat model.
   */
  async getStatus() {
    try {
      const client = new ChromaClient({
        path: this.chromaUrl,
      });

      const [profileCollection, interactionCollection] = await Promise.all([
        client.getCollection({
          name: this.profileCollectionName,
        }),

        client.getCollection({
          name: this.interactionCollectionName,
        }),
      ]);

      const [profileRecords, interactionRecords] = await Promise.all([
        profileCollection.count(),
        interactionCollection.count(),
      ]);

      return {
        success: true,
        chromaUrl: this.chromaUrl,

        collections: {
          drugProfiles: {
            name: this.profileCollectionName,
            records: profileRecords,
            ready: profileRecords > 0,
          },

          drugInteractions: {
            name: this.interactionCollectionName,
            records: interactionRecords,
            ready: interactionRecords > 0,
          },
        },

        ready: profileRecords > 0 && interactionRecords > 0,
      };
    } catch (error) {
      console.error('Chroma status error:', error);

      throw new ServiceUnavailableException({
        message: 'تعذر الاتصال بـ ChromaDB أو الوصول إلى إحدى مجموعات RAG.',

        chromaUrl: this.chromaUrl,

        profileCollection: this.profileCollectionName,

        interactionCollection: this.interactionCollectionName,

        details: this.getErrorMessage(error),
      });
    }
  }

  /**
   * Retrieval-only endpoint.
   *
   * The question is first routed to:
   * - drug profiles,
   * - drug interactions,
   * - or both.
   */
  // async search(dto: AskRagDto) {
  //   const question = this.validateQuestion(dto.question);

  //   const topK = this.resolveTopK(dto.topK);

  //   const route = this.detectQueryRoute(question);

  //   try {
  //     const resources = await this.getResources();

  //     const results = await this.retrieveByRoute(
  //       resources,
  //       question,
  //       topK,
  //       route,
  //     );

  //     return {
  //       question,
  //       route,

  //       collectionsSearched: this.getCollectionsForRoute(route),

  //       embeddingModel: this.embeddingModel,

  //       topKPerCollection: topK,

  //       retrievedDocuments: results.length,

  //       sources: this.mapSources(results),
  //     };
  //   } catch (error) {
  //     this.handleRagError(error, 'retrieval');
  //   }
  // }

  async search(dto: AskRagDto) {
    const question = this.validateQuestion(dto.question);

    const conversationSummary = this.validateConversationSummary(
      dto.conversationSummary,
    );

    const topK = this.resolveTopK(dto.topK);

    /**
     * Important:
     * Routing is based primarily on the CURRENT question.
     *
     * We don't use the previous summary to decide PROFILE vs INTERACTIONS,
     * because the summary may contain old topics.
     */
    const route = this.detectQueryRoute(question);

    /**
     * But the summary IS useful for semantic retrieval,
     * especially for follow-up questions like:
     *
     * "وما تداخلاته؟"
     * "وما موانع استعماله؟"
     */
    const retrievalQuery = this.buildRetrievalQuery(
      question,
      conversationSummary,
    );

    try {
      const resources = await this.getResources();

      const results = await this.retrieveByRoute(
        resources,
        retrievalQuery,
        topK,
        route,
      );

      return {
        question,

        conversationSummary,

        route,

        collectionsSearched: this.getCollectionsForRoute(route),

        embeddingModel: this.embeddingModel,

        topKPerCollection: topK,

        retrievedDocuments: results.length,

        sources: this.mapSources(results),
      };
    } catch (error) {
      this.handleRagError(error, 'retrieval');
    }
  }

  /**
   * Complete RAG flow:
   * question
   * -> route detection
   * -> retrieval from one/both collections
   * -> context construction
   * -> Gemini answer
   */
  // async ask(dto: AskRagDto) {
  //   const question = this.validateQuestion(dto.question);

  //   const topK = this.resolveTopK(dto.topK);

  //   const route = this.detectQueryRoute(question);

  //   try {
  //     const resources = await this.getResources();

  //     const results = await this.retrieveByRoute(
  //       resources,
  //       question,
  //       topK,
  //       route,
  //     );

  //     if (results.length === 0) {
  //       return {
  //         question,
  //         route,
  //         answer: 'لا أعرف. لم أجد معلومات مرتبطة بالسؤال في قاعدة المعرفة.',
  //         retrievedDocuments: 0,
  //         sources: [],
  //       };
  //     }

  //     const context = this.formatContext(results);

  //     const prompt = this.buildPrompt();

  //     const response = await prompt.pipe(resources.model).invoke({
  //       question,
  //       route,
  //       context,
  //     });

  //     return {
  //       question,
  //       route,

  //       answer: this.extractMessageText(response.content),

  //       collectionsSearched: this.getCollectionsForRoute(route),

  //       embeddingModel: this.embeddingModel,

  //       chatModel: this.chatModel,

  //       topKPerCollection: topK,

  //       retrievedDocuments: results.length,

  //       sources: this.mapSources(results),
  //     };
  //   } catch (error) {
  //     this.handleRagError(error, 'generation');
  //   }
  // }

  async ask(dto: AskRagDto) {
    const question = this.validateQuestion(dto.question);

    const conversationSummary = this.validateConversationSummary(
      dto.conversationSummary,
    );

    const topK = this.resolveTopK(dto.topK);

    /**
     * Determine which collection should be searched
     * based on the current user question.
     */
    const route = this.detectQueryRoute(question);

    /**
     * Use conversation memory while searching ChromaDB.
     *
     * This allows follow-up questions such as:
     * "وما تداخلاته؟"
     * "وماذا عن موانع الاستعمال؟"
     */
    const retrievalQuery = this.buildRetrievalQuery(
      question,
      conversationSummary,
    );

    try {
      const resources = await this.getResources();

      const results = await this.retrieveByRoute(
        resources,
        retrievalQuery,
        topK,
        route,
      );

      /**
       * Even when no documents are retrieved,
       * we still call the LLM.
       *
       * Why?
       * Because it must return updatedSummary in addition
       * to the answer.
       *
       * The prompt will force it to answer "لا أعرف"
       * when there is no supporting RAG context.
       */
      const context =
        results.length > 0
          ? this.formatContext(results)
          : 'لم يتم استرجاع أي معلومات مرتبطة بالسؤال من قاعدة المعرفة.';

      const prompt = this.buildPrompt();

      const response = await prompt.pipe(resources.model).invoke({
        question,

        route,

        context,

        conversationSummary:
          conversationSummary ?? 'لا يوجد ملخص سابق. هذه بداية محادثة جديدة.',
      });

      /**
       * Gemini must return:
       *
       * {
       *   "answer": "...",
       *   "updatedSummary": "..."
       * }
       */
      const rawResponse = this.extractMessageText(response.content);

      const parsedResponse = this.parseLlmResponse(rawResponse);

      return {
        answer: parsedResponse.answer,

        updatedSummary: parsedResponse.updatedSummary,

        conversationTitle:"new one",

        route,

        collectionsSearched: this.getCollectionsForRoute(route),

        retrievedDocuments: results.length,

        sources: this.mapSources(results),
      };
    } catch (error) {
      this.handleRagError(error, 'generation');
    }
  }

  /** Initialize resources once and reuse them. */
  private getResources(): Promise<RagResources> {
    if (!this.resourcesPromise) {
      this.resourcesPromise = this.initialize().catch((error: unknown) => {
        this.resourcesPromise = undefined;

        throw error;
      });
    }

    return this.resourcesPromise;
  }

  /** Connect LangChain to both existing Chroma collections. */
  private async initialize(): Promise<RagResources> {
    const apiKey = this.getRequiredEnv('GOOGLE_API_KEY');

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey,
      model: this.embeddingModel,
    });

    const [profileStore, interactionStore] = await Promise.all([
      Chroma.fromExistingCollection(embeddings, {
        collectionName: this.profileCollectionName,
        url: this.chromaUrl,
      }),

      Chroma.fromExistingCollection(embeddings, {
        collectionName: this.interactionCollectionName,
        url: this.chromaUrl,
      }),
    ]);

    const model = new ChatGoogleGenerativeAI({
      apiKey,
      model: this.chatModel,
      temperature: 0,
      maxRetries: 2,
    });

    return {
      profileStore,
      interactionStore,
      model,
    };
  }

  /**
   * Decide which knowledge source should answer the question.
   */
  private detectQueryRoute(question: string): QueryRoute {
    const normalized = this.normalizeText(question);

    const asksForInteractions =
      /(?:تداخل|تداخلات|تعارض|تعارضات|تضارب|تضاربات|يتداخل|يتعارض|يتضارب|يتفاعل|interaction|interactions|interact|drug interaction)/i.test(
        normalized,
      ) ||
      /(?:هل\s*)?(?:يمكن|يجوز|ينفع).{0,100}(?:مع|معا|معاً)|(?:اخذ|أخذ|تناول|استخدام).{0,100}(?:مع|معا|معاً)|taken with|given with|use with|together/i.test(
        normalized,
      );

    const asksForProfile =
      /(?:ماهو|ما هو|ماهي|ما هي|عرف|تعريف|معلومات|الماده الفعاله|المادة الفعالة|تركيب|مكونات|استخدام|استخدامات|استعمال|استعمالات|دواعي|استطباب|استطبابات|جرعه|جرعات|جرعة|تحذير|تحذيرات|اعراض|أعراض|اثار جانبيه|آثار جانبية|موانع|مانع استعمال|الشكل الصيدلاني|شكل صيدلاني|عبوه|عبوة|حجم العبوه|حجم العبوة|dosage|indication|contraindication|warning|side effect|active ingredient|drug information)/i.test(
        normalized,
      );

    if (asksForProfile && asksForInteractions) {
      return 'BOTH';
    }

    if (asksForInteractions) {
      return 'INTERACTIONS_ONLY';
    }

    if (asksForProfile) {
      return 'PROFILE_ONLY';
    }

    /**
     * Unknown question type:
     * search both sources rather than guessing.
     */
    return 'BOTH';
  }

  private async retrieveByRoute(
    resources: RagResources,
    question: string,
    topK: number,
    route: QueryRoute,
  ): Promise<RoutedSearchResult[]> {
    const searches: Array<Promise<RoutedSearchResult[]>> = [];

    if (route !== 'INTERACTIONS_ONLY') {
      searches.push(
        this.searchStore(
          resources.profileStore,
          question,
          topK,
          'DRUG_PROFILES',
        ),
      );
    }

    if (route !== 'PROFILE_ONLY') {
      searches.push(
        this.searchStore(
          resources.interactionStore,
          question,
          topK,
          'DRUG_INTERACTIONS',
        ),
      );
    }

    const groups = await Promise.all(searches);

    const results = this.deduplicateResults(groups.flat());

    /**
     * Chroma returns distance: smaller is better.
     * Sorting is especially useful when BOTH collections were searched.
     */
    return results.sort((first, second) => first.distance - second.distance);
  }

  private async searchStore(
    store: Chroma,
    question: string,
    topK: number,
    collection: KnowledgeCollection,
  ): Promise<RoutedSearchResult[]> {
    const results: SearchResult[] = await store.similaritySearchWithScore(
      question,
      topK,
    );

    return results.map(([document, distance]) => ({
      collection,
      document,
      distance,
    }));
  }

  private deduplicateResults(
    results: RoutedSearchResult[],
  ): RoutedSearchResult[] {
    const seen = new Set<string>();

    const output: RoutedSearchResult[] = [];

    for (const result of results) {
      const metadata = result.document.metadata ?? {};

      const key = [
        result.collection,
        String(metadata.document_kind ?? ''),
        String(metadata.entry_index ?? ''),
        String(metadata.interaction_index ?? ''),
        String(metadata.json_index ?? ''),
        result.document.pageContent,
      ].join('::');

      if (seen.has(key)) {
        continue;
      }

      seen.add(key);
      output.push(result);
    }

    return output;
  }

  private getCollectionsForRoute(route: QueryRoute): string[] {
    if (route === 'PROFILE_ONLY') {
      return [this.profileCollectionName];
    }

    if (route === 'INTERACTIONS_ONLY') {
      return [this.interactionCollectionName];
    }

    return [this.profileCollectionName, this.interactionCollectionName];
  }

  /** Prompt used for final answer generation. */
  //   private buildPrompt(): ChatPromptTemplate {
  //     return ChatPromptTemplate.fromTemplate(`
  // أنت مساعد صيدلاني يعمل ضمن نظام RAG تجريبي يعتمد على مصدرين منفصلين:
  // 1. DRUG_PROFILES لمعلومات الدواء العامة.
  // 2. DRUG_INTERACTIONS لمعلومات التداخلات والتعارضات الدوائية.

  // مسار السؤال الحالي:
  // {route}

  // قواعد إلزامية:
  // 1. أجب اعتماداً على النص المسترجع فقط.
  // 2. لا تستخدم أي معرفة طبية خارج النص المسترجع.
  // 3. لا تعتبر معلومات DRUG_PROFILES دليلاً على وجود أو عدم وجود تداخل دوائي.
  // 4. أي حكم عن تداخل أو تعارض بين دوائين يجب أن يكون مدعوماً صراحة بوثيقة من DRUG_INTERACTIONS.
  // 5. غياب سجل تداخل من النتائج لا يعني طبياً أن الجمع بين الدواءين آمن. إذا لم يوجد سجل صريح يجيب عن سؤال التداخل، أجب: لا أعرف.
  // 6. لا تخترع effect أو severity أو evidence أو action أو mechanism غير موجودة في البيانات.
  // 7. إذا كان الحقل المطلوب فارغاً أو غير مذكور، قل إنه غير مذكور في البيانات.
  // 8. عند وجود معلومات عن التداخل، وضّح الأطراف المتداخلة والتأثير والشدة أو الإجراء فقط إذا كانت هذه القيم موجودة صراحة.
  // 9. أجب باللغة العربية، مع إبقاء أسماء الأدوية والمصطلحات الطبية الأصلية كما وردت في المصدر.
  // 10. اجعل الإجابة مباشرة ومختصرة، إلا إذا طلب المستخدم تفاصيل.
  // 11 . لا تذكر اسماء الملفات DRUG_INTERACTIONS و DRUG_PROFILES في الإجابة.
  // 12.أذكر ضمن الإجابة رقم صفحة المرجع إذا كان موجوداً في البيانات.
  // النص المسترجع:
  // {context}

  // السؤال:
  // {question}
  // `);
  //   }

  private buildPrompt(): ChatPromptTemplate {
    return ChatPromptTemplate.fromTemplate(`
أنت مساعد صيدلاني يعمل ضمن نظام RAG.

لديك مصدران للمعرفة:
1. DRUG_PROFILES ويحتوي معلومات الأدوية العامة.
2. DRUG_INTERACTIONS ويحتوي معلومات التداخلات والتعارضات الدوائية.

أنت أيضاً مسؤول عن المحافظة على سياق المحادثة من خلال إنشاء ملخص محدث قصير يسمى updatedSummary.

==================================================
ذاكرة المحادثة السابقة
==================================================

{conversationSummary}

مهم جداً:
- ذاكرة المحادثة السابقة تستخدم لفهم سياق الحديث والمقصود من الضمائر والأسئلة المتتابعة فقط.
- لا تعتبر conversationSummary مصدراً طبياً موثوقاً بحد ذاته.
- المعلومات الطبية التي ستقدمها في الإجابة الحالية يجب أن تكون مدعومة بالنص المسترجع من قاعدة المعرفة.
- إذا كان السؤال الحالي يحتوي مثلًا على "وما تداخلاته؟" فاستخدم ملخص المحادثة لمعرفة الدواء المقصود.


==================================================
مسار البحث الحالي
==================================================

{route}

==================================================
النص المسترجع من قاعدة المعرفة
==================================================

{context}

==================================================
السؤال الحالي
==================================================

{question}

==================================================
قواعد الإجابة الطبية
==================================================

1. أجب اعتماداً على النص المسترجع فقط.

2. لا تستخدم أي معرفة طبية خارج النص المسترجع.

3. لا تعتبر معلومات DRUG_PROFILES دليلاً على وجود أو عدم وجود تداخل دوائي.

4. أي حكم عن تداخل أو تعارض بين دوائين يجب أن يكون مدعوماً صراحة بمعلومة مسترجعة من DRUG_INTERACTIONS.

5. غياب سجل تداخل من النتائج لا يعني طبياً أن الجمع بين الدواءين آمن.

6. إذا لم توجد معلومات كافية للإجابة عن السؤال، أجب بوضوح:
"لا أعرف بناءً على البيانات المتوفرة."

7. لا تخترع:
- effect
- severity
- evidence
- action
- mechanism
أو أي معلومة غير موجودة في البيانات.

8. إذا كان الحقل المطلوب فارغاً أو غير مذكور، قل إنه غير مذكور في البيانات.

9. عند الحديث عن تداخل دوائي، وضّح عند توفر المعلومات:
- الأطراف المتداخلة
- التأثير
- الشدة
- الدليل
- الإجراء المطلوب

10. أجب  بنفس لغة السؤال، مع إبقاء أسماء الأدوية والمصطلحات الطبية الأصلية كما وردت في البيانات.

11. اجعل الإجابة مباشرة وواضحة، إلا إذا طلب المستخدم تفاصيل.

12. لا تذكر للمستخدم أسماء:
DRUG_INTERACTIONS
DRUG_PROFILES
أو تفاصيل البنية الداخلية لنظام RAG.

13. اذكر رقم صفحة المرجع إذا كان موجوداً في البيانات وكان مفيداً للإجابة.

==================================================
قواعد updatedSummary
==================================================

بعد إنشاء الإجابة الحالية، أنشئ updatedSummary جديداً.

updatedSummary يجب أن يكون ملخصاً قصيراً ومفيداً للمحادثة الحالية بالكامل.

قم ببنائه اعتماداً على:

1. conversationSummary السابق.
2. السؤال الحالي.
3. الإجابة الحالية.

الهدف من updatedSummary هو أن يتم إرساله إليك مرة أخرى مع السؤال القادم لكي تفهم سياق نفس المحادثة دون الحاجة لإرسال جميع الرسائل السابقة.

يجب أن يحافظ updatedSummary على المعلومات المهمة مثل:

- اسم الدواء أو الأدوية التي تدور حولها المحادثة.
- الموضوع الذي يسأل عنه المستخدم.
- المعلومات المهمة التي تم توضيحها.
- التداخلات أو الأدوية الأخرى التي أصبحت جزءاً من سياق المحادثة.
- أي سؤال بقي بدون إجابة واضحة.

لا تكتب تفاصيل غير ضرورية.

لا تنسخ المحادثة كلمة بكلمة.

لا تجعل الملخص طويلاً.

يفضل أن يكون أقل من 1000 حرف.

إذا كانت conversationSummary السابقة غير موجودة، فهذا يعني أن السؤال الحالي هو بداية محادثة جديدة، وأنشئ الملخص من السؤال والإجابة الحالية فقط.

إذا كانت الإجابة الحالية "لا أعرف" أو لم تكن البيانات كافية:
لا تخترع معلومة داخل الملخص.
اذكر فقط أن المستخدم سأل عن هذه المعلومة وأن البيانات المسترجعة لم تكن كافية للإجابة عنها.

==================================================
صيغة الرد الإلزامية
==================================================

يجب أن يكون ردك JSON صالحاً فقط.

ممنوع إضافة أي نص قبل JSON أو بعده.

ممنوع استخدام Markdown.

ممنوع استخدام \`\`\`json.

أعد النتيجة تماماً بهذا الشكل:

{{
  "answer": "الإجابة التي ستظهر للمستخدم",
  "updatedSummary": "الملخص المحدث للمحادثة"
}}
`);
  }

  private formatContext(results: RoutedSearchResult[]): string {
    return results
      .map((result, index) => {
        return [
          `Document ${index + 1}`,
          `Knowledge Collection: ${result.collection}`,
          `Distance: ${result.distance}`,
          result.document.pageContent,
        ].join('\n');
      })
      .join('\n\n---\n\n');
  }

  private mapSources(results: RoutedSearchResult[]): RetrievedSource[] {
    return results.map((result, index) => {
      const metadata = result.document.metadata ?? {};

      const documentKind = String(metadata.document_kind ?? 'unknown');

      const isInteraction = result.collection === 'DRUG_INTERACTIONS';

      return {
        rank: index + 1,

        distance: Number(result.distance.toFixed(6)),

        collection: result.collection,

        source: String(metadata.source ?? 'unknown'),

        documentKind,

        content: result.document.pageContent,

        profile: isInteraction
          ? null
          : {
              jsonIndex: this.numberOrNull(metadata.json_index),

              tradeName: String(metadata.trade_name ?? ''),

              genericName: String(metadata.generic_name ?? ''),

              activeIngredients: String(metadata.active_ingredients ?? ''),

              therapeuticCategory: String(metadata.therapeutic_category ?? ''),

              diseaseCategory: String(metadata.disease_category ?? ''),

              dosageForm: String(metadata.dosage_form ?? ''),

              packSize: String(metadata.pack_size ?? ''),

              pageRef: String(metadata.page_ref ?? ''),
            },

        interaction: !isInteraction
          ? null
          : {
              entryIndex: this.numberOrNull(metadata.entry_index),

              interactionIndex: this.numberOrNull(metadata.interaction_index),

              subjectName: String(metadata.subject_name ?? ''),

              subjectType: String(metadata.entry_type ?? ''),

              relatedName: String(metadata.related_name ?? ''),

              relatedType: String(metadata.related_type ?? ''),

              appliesToScope: String(metadata.applies_to_scope ?? ''),

              appliesToMembers: String(metadata.applies_to_members ?? ''),

              causingEntity: String(metadata.causing_entity ?? ''),

              affectedEntity: String(metadata.affected_entity ?? ''),

              effectType: String(metadata.effect_type ?? ''),

              severity: String(metadata.severity ?? ''),

              evidence: String(metadata.evidence ?? ''),

              actionCategory: String(metadata.action_category ?? ''),

              printedPage: String(metadata.printed_page ?? ''),

              pdfPage: String(metadata.pdf_page ?? ''),

              sourceReference: String(metadata.source_reference ?? ''),
            },
      };
    });
  }

  private validateQuestion(value: unknown): string {
    const question = String(value ?? '').trim();

    if (!question) {
      throw new BadRequestException('question is required');
    }

    if (question.length > 1000) {
      throw new BadRequestException('question must not exceed 1000 characters');
    }

    return question;
  }

  private resolveTopK(value: unknown): number {
    const topK = value === undefined ? this.defaultTopK : Number(value);

    if (!Number.isInteger(topK) || topK < 1 || topK > 10) {
      throw new BadRequestException('topK must be an integer between 1 and 10');
    }

    return topK;
  }

  private normalizeText(value: string): string {
    return value
      .toLowerCase()
      .replace(/[\u064B-\u065F\u0670]/g, '')
      .replace(/[أإآ]/g, 'ا')
      .replace(/ى/g, 'ي')
      .replace(/ؤ/g, 'و')
      .replace(/ئ/g, 'ي')
      .replace(/ة/g, 'ه')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private numberOrNull(value: unknown): number | null {
    const parsed = Number(value);

    return Number.isFinite(parsed) ? parsed : null;
  }

  private extractMessageText(content: unknown): string {
    if (typeof content === 'string') {
      return content.trim();
    }

    if (Array.isArray(content)) {
      return content
        .map((part: unknown) => {
          if (typeof part === 'string') {
            return part;
          }

          if (
            part &&
            typeof part === 'object' &&
            'text' in part &&
            typeof part.text === 'string'
          ) {
            return part.text;
          }

          return '';
        })
        .filter(Boolean)
        .join('\n')
        .trim();
    }

    return String(content ?? '').trim();
  }

  private getRequiredEnv(name: string): string {
    const value = process.env[name]?.trim();

    if (!value) {
      throw new Error('Missing required environment variable: ' + name);
    }

    return value;
  }

  private handleRagError(
    error: unknown,
    stage: 'retrieval' | 'generation',
  ): never {
    console.error(`RAG ${stage} error:`, error);

    const message = this.getErrorMessage(error);

    const normalizedMessage = message.toLowerCase();

    if (
      message.includes('429') ||
      message.includes('RESOURCE_EXHAUSTED') ||
      normalizedMessage.includes('quota')
    ) {
      throw new ServiceUnavailableException({
        message: 'تم تجاوز حصة Gemini مؤقتاً. أعد المحاولة لاحقاً.',
        stage,
        details: message,
      });
    }

    if (
      normalizedMessage.includes('connect') ||
      normalizedMessage.includes('fetch failed') ||
      normalizedMessage.includes('collection')
    ) {
      throw new ServiceUnavailableException({
        message: 'تعذر الاتصال بـ ChromaDB أو الوصول إلى إحدى مجموعات المعرفة.',

        stage,

        chromaUrl: this.chromaUrl,

        profileCollection: this.profileCollectionName,

        interactionCollection: this.interactionCollectionName,

        details: message,
      });
    }

    throw new InternalServerErrorException({
      message: 'فشل تنفيذ اختبار RAG.',
      stage,
      details: message,
    });
  }

  private getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }

    return String(error);
  }

  private buildRetrievalQuery(
    question: string,
    conversationSummary: string | null,
  ): string {
    if (!conversationSummary) {
      return question;
    }

    return [
      'Current user question:',
      question,
      '',
      'Previous conversation context:',
      conversationSummary,
    ].join('\n');
  }

  private validateConversationSummary(value: unknown): string | null {
    if (value === undefined || value === null) {
      return null;
    }

    const summary = String(value).trim();

    if (!summary) {
      return null;
    }

    /**
     * The summary should stay short.
     *
     * This also protects us from receiving an accidentally
     * huge conversation history instead of a summary.
     */
    if (summary.length > 5000) {
      throw new BadRequestException(
        'conversationSummary must not exceed 5000 characters',
      );
    }

    return summary;
  }

  private parseLlmResponse(rawResponse: string): LlmRagResponse {
    let cleaned = rawResponse.trim();

    /**
     * Extra protection in case the model returns a markdown
     * code block even though the prompt explicitly forbids it.
     */
    cleaned = cleaned
      .replace(/^```json\s*/i, '')
      .replace(/^```\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim();

    /**
     * If the model accidentally puts text around the JSON,
     * try to extract the JSON object.
     */
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');

    if (firstBrace !== -1 && lastBrace > firstBrace) {
      cleaned = cleaned.slice(firstBrace, lastBrace + 1);
    }

    try {
      const parsed = JSON.parse(cleaned) as Partial<LlmRagResponse>;

      const answer = String(parsed.answer ?? '').trim();

      const updatedSummary = String(parsed.updatedSummary ?? '').trim();

      if (!answer) {
        throw new Error('LLM response does not contain answer');
      }

      if (!updatedSummary) {
        throw new Error('LLM response does not contain updatedSummary');
      }

      return {
        answer,
        updatedSummary,
      };
    } catch (error) {
      console.error('Invalid structured LLM response:', {
        rawResponse,
        error,
      });

      throw new Error('Gemini returned an invalid structured RAG response.');
    }
  }
}
