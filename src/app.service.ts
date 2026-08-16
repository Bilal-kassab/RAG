// import { Injectable } from '@nestjs/common';
// import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
// import { Chroma } from "@langchain/community/vectorstores/chroma";
// import { RetrievalQAChain } from "@langchain/classic/chains";
// @Injectable()
// export class AppService {
//   getHello(): string {
//     return 'Hello World!';
//   }


//   async ask(question: string) {
//     // 1. استخدام الـ Embeddings المجانية للبحث
//     const embeddings = new OllamaEmbeddings({
//       model: "mxbai-embed-large",
//       baseUrl: "http://localhost:11434",
//     });

//     // 2. تحميل قاعدة البيانات المحلية
//     const vectorStore = await Chroma.fromExistingCollection(embeddings, {
//       collectionName: "langchain",
//       url: "http://localhost:8000", // تأكد من تشغيل ChromaDB
//     });

//     // 3. استخدام نموذج Llama المجاني للرد
//     const model = new ChatOllama({
//       model: "llama3.2:1b",
//       baseUrl: "http://localhost:11434",
//     });

//     const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
//     const response = await chain.call({ query: question });

//     return { answer: response.text };
//   }
// }



// import { Injectable } from "@nestjs/common";
// import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
// import { Chroma } from "@langchain/community/vectorstores/chroma";
// import { ChatPromptTemplate } from "@langchain/core/prompts";
// import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
// import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";

// @Injectable()
// export class AppService {
//   async ask(question: string) {
//     const embeddings = new OllamaEmbeddings({
//       model: "mxbai-embed-large",
//       baseUrl: "http://localhost:11434",
//     });

//     const vectorStore = await Chroma.fromExistingCollection(embeddings, {
//       collectionName: "langchain",
//       url: "http://localhost:8000",
//     });

//     const model = new ChatOllama({
//       model: "llama3.2:1b",
//       baseUrl: "http://localhost:11434",
//     });

//     const prompt = ChatPromptTemplate.fromMessages([
//       ["system", "أجب فقط من السياق. إذا لم تجد الجواب في السياق قل: لا أعرف.\n\nالسياق:\n{context}"],
//       ["human", "{input}"],
//     ]);

//     const combineDocsChain = await createStuffDocumentsChain({
//       llm: model,
//       prompt,
//     });

//     const chain = await createRetrievalChain({
//       retriever: vectorStore.asRetriever(),
//       combineDocsChain,
//     });

//     const result = await chain.invoke({ input: question }); 

//     return { answer: result.answer };
//   }


// }


// import { Injectable } from "@nestjs/common";
// import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
// import { Chroma } from "@langchain/community/vectorstores/chroma";
// import { ChatPromptTemplate } from "@langchain/core/prompts";
// import { formatDocumentsAsString } from "@langchain/classic/util/document";

// @Injectable()
// export class AppService {
//   async ask(question: string) {
//     const embeddings = new OllamaEmbeddings({
//       model: "mxbai-embed-large",
//       baseUrl: "http://localhost:11434",
//     });

//     // ✅ استخدم host/port بدل url (حسب docs)
//     const vectorStore = await Chroma.fromExistingCollection(embeddings, {
//       collectionName: "langchain",
//       url: "http://localhost:8000",
//     });

//     const model = new ChatOllama({
//       model: "llama3.2:1b",
//       baseUrl: "http://localhost:11434",
//       temperature: 0, // يقلل الهلوسة
//     });

//     // ✅ 1) اسحب النتائج أولاً
//     const docsWithScore = await vectorStore.similaritySearchWithScore(question, 4);

//     // ✅ 2) threshold: إذا ما فيه نتائج كافية، رجّع "لا أعرف" بدون ما تسأل الموديل
//     // ملاحظة: في Chroma عادة "الأقل أفضل" (مسافة). جرّب القيم 0.7 إلى 1.2 حسب بياناتك.
//     const GOOD_MAX_DISTANCE = 1.0;
//     const goodDocs = docsWithScore
//       .filter(([_, score]) => score <= GOOD_MAX_DISTANCE)
//       .map(([doc]) => doc);

//     if (goodDocs.length === 0) {
//       return { answer: "لا أعرف" };
//     }

//     // ✅ 3) ابني السياق من الوثائق
//     const context = formatDocumentsAsString(goodDocs);

//     const prompt = ChatPromptTemplate.fromMessages([
//       ["system", "أجب فقط من السياق التالي. إذا لم تجد الجواب في السياق اكتب حرفيًا: لا أعرف. اكتب بالعربية فقط."],
//       ["human", "السؤال: {input}\n\nالسياق:\n{context}"],
//     ]);

//     const resp = await prompt.pipe(model).invoke({
//       input: question,
//       context,
//     });

//     return { answer: String(resp.content).trim() };
//   }
// }





// import { Injectable } from '@nestjs/common';
// import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
// import { Chroma } from "@langchain/community/vectorstores/chroma";
// import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
// import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";


// import { ChatPromptTemplate } from "@langchain/core/prompts";

// @Injectable()
// export class AppService {
//   private readonly apiKey = "AIzaSyBGBPxeF8lPlLUdU4yzZ6qteyk6emBi-EI";
//   // private readonly apiKey = "AIzaSyCK8fOmUIC6E_Wic2RHrYpjbprBcfr3vkE";
  
//   async ask(question: string) {

//     const embeddings = new GoogleGenerativeAIEmbeddings({
//       apiKey: this.apiKey,
//       // modelName: "models/text-embedding-004",
//       modelName: "gemini-embedding-001",
//     });
    
//     const vectorStore = await Chroma.fromExistingCollection(embeddings, {
//       collectionName: "langchain_gemini2",
//       url: "http://localhost:8000",
//     });

//     // 3. Gemini Flash 
//     const model = new ChatGoogleGenerativeAI({
//       apiKey: this.apiKey,
//       model: "gemini-3.1-flash-lite-preview",
//       // model: "models/gemini-2.5-flash",
//       maxRetries: 0,
//       temperature: 0,
//     });

//     const docs = await vectorStore.similaritySearch(question, 2);
//     console.log("Retrieved docs count:", docs.length);
//     // console.log("First doc:", docs[0]?.pageContent);
//     // if (docs.length === 0) {
//     //   return { answer: "لا أعرف" };
//     // }
//     // const prompt = ChatPromptTemplate.fromTemplate(`
//     //   Answer the question based only on the context:
//     //   {context}
//     //   Question: {input}
//     // `);

//     //  const prompt = ChatPromptTemplate.fromTemplate(`
//     //     You must answer ONLY using the context.
//     //     If the answer is not explicitly in the context, reply EXACTLY with: لا أعرف

//     //     Context:
//     //     {context}

//     //     Question: {input}
//     //   `);
//     const prompt = ChatPromptTemplate.fromTemplate(`
// أنت مساعد طبي ضمن نظام RAG.
// You must answer ONLY using the context.
// قواعد صارمة:
// - أجب فقط من السياق.
// - ممنوع استخدام أي معرفة خارجية.
// - إذا لم تكن الإجابة موجودة صراحة في السياق، أجب فقط: لا أعرف
// - أجب باللغة العربية.
// - أعد الإجابة بصيغة Markdown فقط.
// - استخدم هذا التنسيق بالضبط:

// ## الإجابة المختصرة
// نص مختصر من سطر إلى سطرين.

// ## التفاصيل
// - نقطة أولى
// - نقطة ثانية
// - نقطة ثالثة

// ## التحذيرات
// - نقطة أولى
// - نقطة ثانية
// - إذا لم توجد تحذيرات، اكتب: لا توجد تحذيرات مذكورة في السياق.


// السياق:
// {context}

// السؤال:
// {input}
// `);

    
//     const combineDocsChain = await createStuffDocumentsChain({
//       llm: model,
//       prompt,
//     });

    
//     const retrievalChain = await createRetrievalChain({
//       combineDocsChain,
//       retriever: vectorStore.asRetriever(),
//     });

//     console.log("Combine Docs Chain created");
//     try {
//       const response = await retrievalChain.invoke({ input: question });
//       console.log("return successfully");
//       return { answer: response.answer };
//     } catch (error: any) {
//       console.error("Gemini error:", error);

//       return {
//         answer: "حدث ضغط مؤقت على خدمة الذكاء الاصطناعي. حاول مرة أخرى بعد قليل.",
//       };
//     }
//   }

//   // async ask(question: string) {
//   //   const vectorStore = await this.vectorStorePromise;

//   //   const docs = await vectorStore.similaritySearch(question, 3);
//   //   // ... تابع السلسلة
//   //   console.log("Retrieved documents:", docs);
//   //   const r = await this.model.invoke("قل hello");
//   //   return { answer: String(r.content) };
//   // }
// }


import {
  Injectable,
  InternalServerErrorException,
} from '@nestjs/common';

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from '@langchain/google-genai';

import {
  Chroma,
} from '@langchain/community/vectorstores/chroma';

import {
  Document,
} from '@langchain/core/documents';

import {
  ChatPromptTemplate,
} from '@langchain/core/prompts';

import {
  ChromaClient,
  type Collection,
  type Metadata,
} from 'chromadb';


type QueryRoute =
  | 'PROFILE_ONLY'
  | 'INTERACTIONS_ONLY'
  | 'BOTH';


type InteractionMetadata = Metadata & {
  source?: string;
  document_kind?: string;

  subject_key?: string;
  subject_name?: string;
  subject_aliases?: string;

  related_key?: string;
  related_name?: string;
  related_aliases?: string;
};


type RagResources = {
  profileStore: Chroma;
  interactionStore: Chroma;

  interactionCollection: Collection;

  interactionAliasMap: Map<
    string,
    Set<string>
  >;

  model: ChatGoogleGenerativeAI;
};


@Injectable()
export class AppService {
  private readonly chromaUrl =
    process.env.CHROMA_URL
    ?? 'http://localhost:8000';

  private readonly profileCollectionName =
    process.env.DRUG_PROFILE_COLLECTION
    ?? 'drug_profiles';

  private readonly interactionCollectionName =
    process.env.DRUG_INTERACTION_COLLECTION
    ?? 'drug_interactions';

  private readonly embeddingModel =
    process.env.GEMINI_EMBEDDING_MODEL
    ?? 'gemini-embedding-001';

  private readonly chatModel =
    process.env.GEMINI_CHAT_MODEL
    ?? 'gemini-3.1-flash-lite-preview';

  private resourcesPromise?: Promise<
    RagResources
  >;


  async ask(question: string) {
    const cleanQuestion = String(
      question ?? '',
    ).trim();

    if (!cleanQuestion) {
      return {
        answer: 'يرجى إرسال سؤال غير فارغ.',
        route: null,
        sources: [],
      };
    }

    try {
      const resources =
        await this.getResources();

      const route =
        this.detectQueryRoute(
          cleanQuestion,
        );

      const [
        profileDocuments,
        interactionDocuments,
      ] = await Promise.all([
        route === 'INTERACTIONS_ONLY'
          ? Promise.resolve([])
          : this.searchProfiles(
              resources.profileStore,
              cleanQuestion,
            ),

        route === 'PROFILE_ONLY'
          ? Promise.resolve([])
          : this.searchInteractions(
              resources,
              cleanQuestion,
            ),
      ]);

      const documents =
        this.deduplicateDocuments([
          ...profileDocuments,
          ...interactionDocuments,
        ]);

      if (documents.length === 0) {
        return {
          answer:
            'لا أعرف. لم أجد معلومات مناسبة '
            + 'في قاعدة المعرفة الحالية.',

          route,

          sources: [],
        };
      }

      const context =
        this.formatContext(
          documents,
        );

      const prompt =
        this.buildAnswerPrompt();

      const response =
        await prompt
          .pipe(resources.model)
          .invoke({
            question: cleanQuestion,
            context,
          });

      return {
        answer:
          this.extractMessageText(
            response.content,
          ),

        route,

        sources:
          this.buildSources(
            documents,
          ),
      };
    } catch (error) {
      console.error(
        'RAG error:',
        error,
      );

      throw new InternalServerErrorException(
        'تعذر تنفيذ عملية البحث '
        + 'في قاعدة المعرفة حالياً.',
      );
    }
  }


  private getResources(): Promise<RagResources> {
    this.resourcesPromise ??=
      this.initialize();

    return this.resourcesPromise;
  }


  private async initialize(): Promise<RagResources> {
    // const apiKey =
    //   this.getRequiredEnv(
    //     'GOOGLE_API_KEY',
    //   );
    const apiKey ="AIzaSyCK8fOmUIC6E_Wic2RHrYpjbprBcfr3vkE";

    const embeddings =
      new GoogleGenerativeAIEmbeddings({
        apiKey,

        model:
          this.embeddingModel,
      });

    const [
      profileStore,
      interactionStore,
    ] = await Promise.all([
      Chroma.fromExistingCollection(
        embeddings,
        {
          collectionName:
            this.profileCollectionName,

          url:
            this.chromaUrl,
        },
      ),

      Chroma.fromExistingCollection(
        embeddings,
        {
          collectionName:
            this.interactionCollectionName,

          url:
            this.chromaUrl,
        },
      ),
    ]);

    const chromaClient =
      new ChromaClient({
        path:
          this.chromaUrl,
      });

    const interactionCollection =
      await chromaClient.getCollection({
        name:
          this.interactionCollectionName,
      });

    const interactionAliasMap =
      await this.loadInteractionAliasMap(
        interactionCollection,
      );

    const model =
      new ChatGoogleGenerativeAI({
        apiKey,

        model:
          this.chatModel,

        temperature: 0,

        maxRetries: 2,
      });

    return {
      profileStore,

      interactionStore,

      interactionCollection,

      interactionAliasMap,

      model,
    };
  }


  private detectQueryRoute(
    question: string,
  ): QueryRoute {
    const normalized =
      this.normalizeText(
        question,
      );

    const asksForInteractions =
      /(تداخل|تداخلات|تعارض|تعارضات|تضارب|تضاربات|يضارب|يتضارب|يتعارض|يتفاعل|interaction|interactions|interact)/i
        .test(
          normalized,
        );

    const asksForProfile =
      /(ماهو|ما هو|عرف|تعريف|معلومات|استخدام|استخدامات|استعمال|استعمالات|دواعي|جرعه|جرعات|تحذير|تحذيرات|اعراض|اثار جانبيه|موانع)/i
        .test(
          normalized,
        );

    if (
      asksForProfile
      && asksForInteractions
    ) {
      return 'BOTH';
    }

    if (asksForInteractions) {
      return 'INTERACTIONS_ONLY';
    }

    if (asksForProfile) {
      return 'PROFILE_ONLY';
    }

    /*
     * عندما يكون السؤال غير واضح،
     * يتم البحث في المصدرين.
     */
    return 'BOTH';
  }


  private async searchProfiles(
    profileStore: Chroma,
    question: string,
  ): Promise<Document[]> {
    return profileStore
      .similaritySearch(
        question,
        4,
      );
  }


  private async searchInteractions(
    resources: RagResources,
    question: string,
  ): Promise<Document[]> {
    const mentionedDrugKeys =
      this.resolveMentionedDrugKeys(
        question,

        resources
          .interactionAliasMap,
      );

    /*
     * عندما يذكر المستخدم دواءين أو أكثر:
     * نبحث عن العلاقة المباشرة فقط.
     *
     * لا نعيد علاقات مشابهة تخص أدوية أخرى.
     */
    if (
      mentionedDrugKeys.length >= 2
    ) {
      const directPairDocuments =
        await this.getDirectPairDocuments(
          resources
            .interactionCollection,

          mentionedDrugKeys,
        );

      if (
        directPairDocuments.length > 0
      ) {
        return directPairDocuments;
      }

      return [
        this
          .buildNoInteractionRecordDocument(
            'لم يتم العثور على سجل تضارب '
            + 'مباشر بين الأدوية المذكورة '
            + 'في ملف JSON.',
          ),
      ];
    }

    /*
     * عندما يذكر المستخدم دواءً واحداً:
     * نجلب ملخص جميع تضارباته.
     */
    if (
      mentionedDrugKeys.length === 1
    ) {
      const summaryDocuments =
        await this
          .getDocumentsByMetadata(
            resources
              .interactionCollection,

            {
              $and: [
                {
                  document_kind:
                    'interaction_summary',
                },
                {
                  subject_key:
                    mentionedDrugKeys[0],
                },
              ],
            },
          );

      if (
        summaryDocuments.length > 0
      ) {
        return summaryDocuments;
      }

      /*
       * في حال لم يوجد ملخص،
       * نحاول جلب العلاقات الفردية.
       */
      const pairDocuments =
        await this
          .getDocumentsByMetadata(
            resources
              .interactionCollection,

            {
              $and: [
                {
                  document_kind:
                    'interaction_pair',
                },
                {
                  subject_key:
                    mentionedDrugKeys[0],
                },
              ],
            },
          );

      if (
        pairDocuments.length > 0
      ) {
        return pairDocuments;
      }

      return [
        this
          .buildNoInteractionRecordDocument(
            'لم يتم العثور على تضاربات '
            + 'مسجلة للدواء المذكور '
            + 'في ملف JSON.',
          ),
      ];
    }

    /*
     * إذا لم نتعرف على اسم الدواء:
     * نستخدم البحث الدلالي كخيار احتياطي.
     *
     * يحدث هذا مثلاً عند وجود خطأ إملائي
     * أو عند نسيان إضافة alias داخل JSON.
     */
    return resources
      .interactionStore
      .similaritySearch(
        question,
        6,
      );
  }


  private async getDirectPairDocuments(
    collection: Collection,
    mentionedDrugKeys: string[],
  ): Promise<Document[]> {
    const documents: Document[] = [];

    for (
      let left = 0;
      left < mentionedDrugKeys.length;
      left += 1
    ) {
      for (
        let right = left + 1;
        right < mentionedDrugKeys.length;
        right += 1
      ) {
        const subjectKey =
          mentionedDrugKeys[left];

        const relatedKey =
          mentionedDrugKeys[right];

        const pairDocuments =
          await this
            .getDocumentsByMetadata(
              collection,

              {
                $and: [
                  {
                    document_kind:
                      'interaction_pair',
                  },
                  {
                    subject_key:
                      subjectKey,
                  },
                  {
                    related_key:
                      relatedKey,
                  },
                ],
              },
            );

        documents.push(
          ...pairDocuments,
        );
      }
    }

    return this
      .deduplicateDocuments(
        documents,
      );
  }


  private async getDocumentsByMetadata(
    collection: Collection,
    where: Record<string, unknown>,
  ): Promise<Document[]> {
    const result =
      await collection.get({
        where:
          where as never,

        include: [
          'documents',
          'metadatas',
        ],
      });

    const documents: Document[] = [];

    for (
      let index = 0;
      index < result.ids.length;
      index += 1
    ) {
      const pageContent =
        result.documents?.[index]
        ?? '';

      if (!pageContent) {
        continue;
      }

      documents.push(
        new Document({
          pageContent,

          metadata:
            (
              result.metadatas?.[index]
              ?? {}
            ) as InteractionMetadata,
        }),
      );
    }

    return documents;
  }


  private async loadInteractionAliasMap(
    collection: Collection,
  ): Promise<
    Map<string, Set<string>>
  > {
    const aliasMap =
      new Map<
        string,
        Set<string>
      >();

    const total =
      await collection.count();

    const pageSize = 500;

    for (
      let offset = 0;
      offset < total;
      offset += pageSize
    ) {
      const result =
        await collection.get({
          limit:
            pageSize,

          offset,

          include: [
            'metadatas',
          ],
        });

      for (
        const rawMetadata
        of result.metadatas ?? []
      ) {
        const metadata =
          (
            rawMetadata
            ?? {}
          ) as InteractionMetadata;

        const subjectKey =
          String(
            metadata.subject_key
            ?? '',
          ).trim();

        if (!subjectKey) {
          continue;
        }

        const names = [
          String(
            metadata.subject_name
            ?? '',
          ),

          ...this.parseAliases(
            String(
              metadata.subject_aliases
              ?? '',
            ),
          ),
        ];

        for (const name of names) {
          const normalizedAlias =
            this.normalizeText(
              name,
            );

          if (!normalizedAlias) {
            continue;
          }

          const keys =
            aliasMap.get(
              normalizedAlias,
            )
            ?? new Set<string>();

          keys.add(
            subjectKey,
          );

          aliasMap.set(
            normalizedAlias,
            keys,
          );
        }
      }
    }

    return aliasMap;
  }


  private resolveMentionedDrugKeys(
    question: string,

    aliasMap: Map<
      string,
      Set<string>
    >,
  ): string[] {
    const normalizedQuestion =
      ` ${this.normalizeText(question)} `;

    const matches: Array<{
      alias: string;

      keys: Set<string>;
    }> = [];

    for (
      const [
        alias,
        keys,
      ]
      of aliasMap.entries()
    ) {
      if (
        alias.length < 2
      ) {
        continue;
      }

      if (
        normalizedQuestion.includes(
          ` ${alias} `,
        )
      ) {
        matches.push({
          alias,
          keys,
        });
      }
    }

    /*
     * الأسماء الأطول أولاً.
     * هذا يقلل مشكلة تطابق اسم قصير
     * داخل اسم أطول.
     */
    matches.sort(
      (first, second) =>
        second.alias.length
        - first.alias.length,
    );

    const output: string[] = [];

    for (
      const match
      of matches
    ) {
      for (
        const key
        of match.keys
      ) {
        if (
          !output.includes(key)
        ) {
          output.push(
            key,
          );
        }
      }
    }

    return output;
  }


  private parseAliases(
    value: string,
  ): string[] {
    try {
      const parsed =
        JSON.parse(
          value,
        );

      if (
        !Array.isArray(parsed)
      ) {
        return [];
      }

      return parsed
        .map(
          (item) =>
            String(item).trim(),
        )
        .filter(Boolean);
    } catch {
      return value
        .split('|')
        .map(
          (item) =>
            item.trim(),
        )
        .filter(Boolean);
    }
  }


  private buildNoInteractionRecordDocument(
    message: string,
  ): Document {
    return new Document({
      pageContent: [
        'Source: drug_interactions.json',

        'Lookup Status: '
        + 'no_reviewed_record_found',

        `Message: ${message}`,

        'Important: غياب السجل '
        + 'من قاعدة المعرفة لا يثبت طبياً '
        + 'عدم وجود تضارب.',
      ].join('\n'),

      metadata: {
        source:
          'drug_interactions.json',

        document_kind:
          'interaction_lookup_status',
      },
    });
  }


    private buildAnswerPrompt():
    ChatPromptTemplate {
    return ChatPromptTemplate
      .fromTemplate(`
أجب عن السؤال اعتماداً على النص المسترجع فقط.

قواعد إلزامية:
1. استخدم فقط المعلومات المذكورة صراحة في النص المسترجع.
2. لا تستخدم معرفتك العامة ولا تستنتج معلومات غير مكتوبة.
3. إذا لم يحتوِ النص المسترجع على إجابة صريحة وكاملة للسؤال، أعد هذه العبارة فقط دون أي إضافة:
لا أعرف.
4. لا تذكر قاعدة المعرفة أو السياق أو الملفات أو عملية البحث.
5. لا تستخدم Markdown.
6. لا تستخدم عناوين أو قوائم أو رموز مثل # أو * أو -.
7. أجب بنص عربي عادي ومباشر.
8. اجعل الإجابة مختصرة ومفيدة. استخدم ثلاث جمل كحد أقصى، إلا إذا طلب المستخدم تفاصيل إضافية.
9. لا تضف مقدمة مثل "بناءً على المعلومات المتاحة".
10. لا تقدم رأياً طبياً أو توصية علاجية من عندك.

النص المسترجع:
{context}

السؤال:
{question}
`);
  }


  private formatContext(
    documents: Document[],
  ): string {
    return documents
      .map(
        (
          document,
          index,
        ) => {
          const source =
            String(
              document
                .metadata
                .source
              ?? 'unknown',
            );

          const kind =
            String(
              document
                .metadata
                .document_kind
              ?? 'unknown',
            );

          return [
            `### Context ${index + 1}`,

            `Source: ${source}`,

            `Kind: ${kind}`,

            document.pageContent,
          ].join('\n');
        },
      )
      .join('\n\n');
  }


  private buildSources(
    documents: Document[],
  ) {
    return documents.map(
      (document) => ({
        source:
          String(
            document
              .metadata
              .source
            ?? 'unknown',
          ),

        kind:
          String(
            document
              .metadata
              .document_kind
            ?? 'unknown',
          ),

        subject:
          String(
            document
              .metadata
              .subject_name
            ?? '',
          ),

        related:
          String(
            document
              .metadata
              .related_name
            ?? '',
          ),
      }),
    );
  }


  private deduplicateDocuments(
    documents: Document[],
  ): Document[] {
    const seen =
      new Set<string>();

    return documents.filter(
      (document) => {
        const source =
          String(
            document
              .metadata
              .source
            ?? '',
          );

        const kind =
          String(
            document
              .metadata
              .document_kind
            ?? '',
          );

        const key =
          `${source}::`
          + `${kind}::`
          + document.pageContent;

        if (
          seen.has(key)
        ) {
          return false;
        }

        seen.add(
          key,
        );

        return true;
      },
    );
  }


  private normalizeText(
    value: string,
  ): string {
    return String(
      value ?? '',
    )
      .normalize('NFKD')
      .toLowerCase()
      .replace(
        /[\u064B-\u065F\u0670]/g,
        '',
      )
      .replace(
        /[أإآ]/g,
        'ا',
      )
      .replace(
        /ى/g,
        'ي',
      )
      .replace(
        /ؤ/g,
        'و',
      )
      .replace(
        /ئ/g,
        'ي',
      )
      .replace(
        /ة/g,
        'ه',
      )
      .replace(
        /[^\p{L}\p{N}]+/gu,
        ' ',
      )
      .trim();
  }


  private extractMessageText(
    content: unknown,
  ): string {
    if (
      typeof content === 'string'
    ) {
      return content.trim();
    }

    return JSON.stringify(
      content,
    );
  }


  private getRequiredEnv(
    name: string,
  ): string {
    const value =
      process.env[name]?.trim();

    if (!value) {
      throw new Error(
        'Missing required environment '
        + `variable: ${name}`,
      );
    }

    return value;
  }
}