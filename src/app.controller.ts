import { Body, Controller, Get, Param, Post, Query } from '@nestjs/common';
import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  // @Get()
  // getHello(): string {
  //   return this.appService.getHello();
  // }

 @Post("ask")
  async askQuestion(@Body("question") question: string) {
    // console.log(question);
    return this.appService.ask(question);
  }
  
}

