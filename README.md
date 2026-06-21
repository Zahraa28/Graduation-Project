---
title: CarID — AI Car Identifier
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# CarID — AI Car Identifier

An intelligent car identification system that identifies cars from images using:

- **CLIP + FAISS** — visual similarity search over 25,000+ car images
- **CarNet** — brand/model/generation detection
- **Serper** — real-time web enrichment for every result
- **Groq (LLaMA 3)** — human-like conversational responses

## How to use

1. Upload a car photo
2. Get the make, model, generation, year, pros/cons and a human-like summary
3. Ask follow-up questions via the chat

## Environment variables required

Set these in your Space's **Settings → Repository secrets**:

| Variable | Source |
|---|---|
| `GROQ_API_KEY` | console.groq.com (free) |
| `SERPER_API_KEY` | serper.dev (free tier) |
