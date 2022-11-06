#STAGE 1
FROM node:14-alpine AS build
WORKDIR /usr/src/app
COPY package.json package-lock.json ./
RUN npm install
RUN npm install @angular/cdk leaflet leaflet.markercluster

COPY . .
RUN npm run build

#STAGE 2
FROM nginx:1.17.1-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY --from=build /usr/src/app/dist/map-optim /usr/share/nginx/html