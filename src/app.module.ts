import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';

import { AppController } from './app.controller';
import { AppService } from './app.service';
import { RagModule } from './rag/rag.module';
import { join } from 'path';
// import * as admin from 'firebase-admin';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,

      // Looks for .env beside package.json
      envFilePath: '.env',
    }),

    RagModule,
  ],

  controllers: [
    AppController,
  ],

  providers: [
    AppService,
  ],
})
export class AppModule {
  constructor() {}

//     var admin = require("firebase-admin");

// var serviceAccount = require("path/to/serviceAccountKey.json");

// admin.initializeApp({
//   credential: admin.credential.cert(serviceAccount)
// });


    // Build the path of the Firebase service account file.
  //   const serviceAccountPath = join(
  //     process.cwd(),
  //     'secrets',
  //     'firebase-service-account.json',
  //   );

  //   // Load Firebase credentials.
  //   const serviceAccount = require(serviceAccountPath);

  //   // Initialize Firebase only once.
  //   if (!admin.apps.length) {
  //     admin.initializeApp({
  //       credential: admin.credential.cert(
  //         serviceAccount,
  //       ),
  //     });
  //   }

  // }
}
  