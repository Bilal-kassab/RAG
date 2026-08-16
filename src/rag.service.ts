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
import { AskRagDto } from './rag/dto/ask-rag.dto';


type SearchResult = [document: Document, distance: number];

type RagResources = {
  vectorStore: Chroma;

  model: ChatGoogleGenerativeAI;
};

type RetrievedSource = {
  rank: number;

  distance: number;

  source: string;

  documentKind: string;

  jsonIndex: number | null;

  tradeName: string;

  genericName: string;

  activeIngredients: string;

  therapeuticCategory: string;

  diseaseCategory: string;

  dosageForm: string;

  packSize: string;

  pageRef: string;

  content: string;
};

@Injectable()
export class RagService {
  /**
   * Chroma HTTP server.
   */
  private readonly chromaUrl =
    process.env.EXTRACTED_DRUGS_CHROMA_URL ?? 'http://localhost:8001';

  /**
   * Must match the collection
   * created by Python.
   */
  private readonly collectionName =
    process.env.EXTRACTED_DRUGS_COLLECTION ?? 'extracted_drug_rows';

  /**
   * Must be exactly the same
   * embedding model used by Python.
   */
  private readonly embeddingModel =
    process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-001';

  private readonly chatModel =
    process.env.GEMINI_CHAT_MODEL ?? 'gemini-3.1-flash-lite';

  private readonly defaultTopK = Number(process.env.RAG_TOP_K ?? '4');

  /**
   * Initialize resources once
   * and reuse them.
   */
  private resourcesPromise?: Promise<RagResources>;

  /**
   * Test Chroma connection
   * without calling Gemini.
   */
  async getStatus() {
    try {
      const client = new ChromaClient({
        path: this.chromaUrl,
      });

      const collection = await client.getCollection({
        name: this.collectionName,
      });

      const records = await collection.count();

      return {
        success: true,

        chromaUrl: this.chromaUrl,

        collection: this.collectionName,

        records,

        ready: records > 0,
      };
    } catch (error) {
      console.error('Chroma status error:', error);

      throw new ServiceUnavailableException({
        message:
          'تعذر الاتصال بقاعدة ChromaDB ' + 'أو لم يتم العثور على المجموعة.',

        chromaUrl: this.chromaUrl,

        collection: this.collectionName,

        details: this.getErrorMessage(error),
      });
    }
  }

  /**
   * Retrieval only.
   *
   * This endpoint does NOT ask Gemini
   * to generate an answer.
   */
  async search(dto: AskRagDto) {
    const question = this.validateQuestion(dto.question);

    const topK = this.resolveTopK(dto.topK);

    try {
      const resources = await this.getResources();

      const results = await resources.vectorStore.similaritySearchWithScore(
        question,
        topK,
      );

      return {
        question,

        collection: this.collectionName,

        embeddingModel: this.embeddingModel,

        topK,

        retrievedDocuments: results.length,

        sources: this.mapSources(results),
      };
    } catch (error) {
      this.handleRagError(error, 'retrieval');
    }
  }

  /**
   * Complete RAG flow:
   *
   * question
   * -> embedding
   * -> Chroma search
   * -> context
   * -> Gemini
   * -> answer
   */
  async ask(dto: AskRagDto) {
    const question = this.validateQuestion(dto.question);

    const topK = this.resolveTopK(dto.topK);

    try {
      const resources = await this.getResources();

      const results = await resources.vectorStore.similaritySearchWithScore(
        question,
        topK,
      );

      if (results.length === 0) {
        return {
          question,

          answer:
            'لا أعرف. لم أجد معلومات ' +
            'مرتبطة بالسؤال ' +
            'في قاعدة المعرفة.',

          collection: this.collectionName,

          retrievedDocuments: 0,

          sources: [],
        };
      }

      const context = this.formatContext(results);

      const prompt = this.buildPrompt();

      const response = await prompt.pipe(resources.model).invoke({
        question,
        context,
      });

      return {
        question,

        answer: this.extractMessageText(response.content),

        collection: this.collectionName,

        embeddingModel: this.embeddingModel,

        chatModel: this.chatModel,

        topK,

        retrievedDocuments: results.length,

        sources: this.mapSources(results),
      };
    } catch (error) {
      this.handleRagError(error, 'generation');
    }
  }

  /**
   * Initialize only once.
   */
  private getResources(): Promise<RagResources> {
    if (!this.resourcesPromise) {
      this.resourcesPromise = this.initialize().catch((error: unknown) => {
        /*
         * Retry initialization
         * on the next request.
         */
        this.resourcesPromise = undefined;

        throw error;
      });
    }

    return this.resourcesPromise;
  }

  /**
   * Connect NestJS to the
   * existing Chroma collection.
   */
  private async initialize(): Promise<RagResources> {
    const apiKey = this.getRequiredEnv('GOOGLE_API_KEY');

    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey,

      model: this.embeddingModel,
    });

    const vectorStore = await Chroma.fromExistingCollection(embeddings, {
      collectionName: this.collectionName,

      url: this.chromaUrl,
    });

    const model = new ChatGoogleGenerativeAI({
      apiKey,

      model: this.chatModel,

      temperature: 0,

      maxRetries: 2,
    });

    return {
      vectorStore,

      model,
    };
  }

  /**
   * Prompt for final answer generation.
   */
  private buildPrompt(): ChatPromptTemplate {
    return ChatPromptTemplate.fromTemplate(`
أنت مساعد صيدلاني يعمل ضمن نظام RAG تجريبي.

قواعد إلزامية:

1. أجب اعتماداً على النص المسترجع فقط.

2. لا تستخدم أي معرفة طبية خارج النص المسترجع.

3. إذا لم توجد إجابة صريحة في النص المسترجع، أجب فقط:
لا أعرف.

4. لا تخترع جرعة أو تحذيراً أو مانع استعمال أو استطباباً غير موجود.

5. إذا كان الحقل المطلوب غير مذكور في البيانات، قل إنه غير مذكور في البيانات.

6. أجب باللغة العربية، مع إبقاء أسماء الأدوية والمصطلحات الأصلية كما وردت.

7. إذا كانت عدة سجلات تخص نفس الاسم التجاري، ميّز بينها حسب المادة الفعالة أو الشكل الصيدلاني أو الصفحة عندما يكون ذلك مفيداً.

8. اجعل الإجابة مباشرة ومختصرة، إلا إذا طلب المستخدم تفاصيل.


النص المسترجع:

{context}


السؤال:

{question}
`);
  }

  /**
   * Build Gemini context.
   */
  private formatContext(results: SearchResult[]): string {
    return results
      .map(([document, distance], index) => {
        return [
          `Document ${index + 1}`,

          `Distance: ${distance}`,

          document.pageContent,
        ].join('\n');
      })
      .join('\n\n---\n\n');
  }

  /**
   * Format retrieval information
   * for API testing.
   */
  private mapSources(results: SearchResult[]): RetrievedSource[] {
    return results.map(([document, distance], index) => {
      const metadata = document.metadata ?? {};

      const jsonIndexValue = Number(metadata.json_index);

      return {
        rank: index + 1,

        distance: Number(distance.toFixed(6)),

        source: String(metadata.source ?? 'unknown'),

        documentKind: String(metadata.document_kind ?? 'unknown'),

        jsonIndex: Number.isFinite(jsonIndexValue) ? jsonIndexValue : null,

        tradeName: String(metadata.trade_name ?? ''),

        genericName: String(metadata.generic_name ?? ''),

        activeIngredients: String(metadata.active_ingredients ?? ''),

        therapeuticCategory: String(metadata.therapeutic_category ?? ''),

        diseaseCategory: String(metadata.disease_category ?? ''),

        dosageForm: String(metadata.dosage_form ?? ''),

        packSize: String(metadata.pack_size ?? ''),

        pageRef: String(metadata.page_ref ?? ''),

        content: document.pageContent,
      };
    });
  }

  private validateQuestion(value: unknown): string {
    const question = String(value ?? '').trim();

    if (!question) {
      throw new BadRequestException('question is required');
    }

    if (question.length > 1000) {
      throw new BadRequestException(
        'question must not exceed ' + '1000 characters',
      );
    }

    return question;
  }

  private resolveTopK(value: unknown): number {
    const topK = value === undefined ? this.defaultTopK : Number(value);

    if (!Number.isInteger(topK) || topK < 1 || topK > 10) {
      throw new BadRequestException(
        'topK must be an integer ' + 'between 1 and 10',
      );
    }

    return topK;
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
      throw new Error('Missing required environment ' + `variable: ${name}`);
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
        message: 'تم تجاوز حصة Gemini مؤقتاً. ' + 'أعد المحاولة لاحقاً.',

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
        message:
          'تعذر الاتصال بـ ChromaDB ' + 'أو الوصول إلى المجموعة المطلوبة.',

        stage,

        chromaUrl: this.chromaUrl,

        collection: this.collectionName,

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
}
