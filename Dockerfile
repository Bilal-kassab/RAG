FROM node:22-bookworm-slim AS builder

WORKDIR /app

COPY package*.json ./

RUN npm ci --no-audit --no-fund

COPY . .

RUN npm run build

RUN npm prune --omit=dev --no-audit


FROM node:22-bookworm-slim AS production

WORKDIR /app

ENV NODE_ENV=production

COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist

USER node

EXPOSE 3000

CMD ["node", "dist/main.js"]