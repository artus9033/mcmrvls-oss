#!/bin/sh

cd /frontend

# since this is a mounted volume, there is no chance to install everything at build time
corepack install
pnpm install

pnpm start
