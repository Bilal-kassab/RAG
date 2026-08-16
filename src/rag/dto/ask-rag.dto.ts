export class AskRagDto {
  question!: string;

  /**
   * Short memory of the current conversation.
   *
   * null / undefined => this is a new conversation.
   */
  conversationSummary?: string | null;

  /**
   * Optional.
   * Mainly useful while testing the RAG endpoint.
   */
  topK?: number;
}
