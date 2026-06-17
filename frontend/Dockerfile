FROM node:26-alpine AS builder
ARG NPM_CONFIG_REGISTRY
WORKDIR /app

COPY package*.json ./
ARG VITE_API_BASE_URL
RUN npm install --registry $NPM_CONFIG_REGISTRY
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
