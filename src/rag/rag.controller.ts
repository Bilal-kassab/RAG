import {
  Body,
  Controller,
  Get,
  HttpCode,
  HttpStatus,
  Post,
} from '@nestjs/common';

import { AskRagDto } from './dto/ask-rag.dto';

import { RagService } from './rag.service';

@Controller('rag-test')
export class RagController {
  constructor(private readonly ragService: RagService) {}

  /**
   * Check ChromaDB without calling Gemini.
   */
  @Get('status')
  getStatus() {
    return this.ragService.getStatus();
  }

  /**
   * Test document retrieval only.
   */
  @Post('search')
  @HttpCode(HttpStatus.OK)
  search(
    @Body()
    dto: AskRagDto,
  ) {
    return this.ragService.search(dto);
  }

  /**
   * Test the complete RAG flow.
   */
  @Post('ask')
  @HttpCode(HttpStatus.OK)
  ask(
    @Body()
    dto: AskRagDto,
  ) {
    return this.ragService.ask(dto);
  }
}
