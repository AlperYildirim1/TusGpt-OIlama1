from vectordb import get_embeddings, query_database, get_client, rerank
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ollama import AsyncClient  # Replace OpenAI with Ollama
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import tiktoken
import json

load_dotenv()

router = APIRouter()

class QueryParams(BaseModel):
    collection: str
    prompt: str
    top_k: int

# Function to count tokens (approximation for Ollama models)
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

@router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Initialize Ollama async client
    client = AsyncClient(host="http://93.127.138.13:11434")  # Ollama's default local server

    try:
        while True:
            data = await websocket.receive_text()
            query_params = QueryParams.parse_raw(data)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Sen bir Retrieval-Augmented Generation (RAG) sisteminde asistansın. "
                        "Görevin, verilen metin parçalarına dayanarak kullanıcının sorgusuna doğru yanıtlar sağlamak. "
                        "Yanıtında, uygun yerlerde sağlanan metne atıfta bulun."
                        "Sana getirilen metinler soruyla kabaca en alakalı olabilecek metinler ama kesinlikle soruyla alakalı demek değil."
                        "Senin bu metinlerden hangisinin soruyla alakalı olabileceğini göz önünde bulundurarak tıbbi bir cevap üretmen gerekiyor."
                    ),
                }
            ]   

            await get_client()
            text = await query_database(query_params.collection, query_params.prompt, query_params.top_k)
            print(text)
            user_input = query_params.prompt  # The user query
            reranked = await rerank(user_input, text)
            print("********* Retrieved Texts *********")
            for i in text:
                print(i)

            print("********* Reranked Texts *********")
            for i in reranked:
                print(i)

            top5 = reranked[:10]  # Top 10 results (adjust if needed)
            print("********* Top 10 Ranked Results *********")
            for i in top5:
                print(i)

            top5_indexes = [item['index'] for item in top5]
            top5_texts = [text[i] for i in top5_indexes]

            # Structure the content clearly: query + retrieved context
            full_content = (
                f"User Query: {user_input}\n\n"
                "Retrieved Context:\n" + "\n".join(f"- {txt}" for txt in top5_texts)
            )

            messages.append({"role": "user", "content": full_content})
            print(full_content)

            # Calculate input tokens
            input_text = "".join([msg["content"] for msg in messages])
            input_tokens = count_tokens(input_text)
            print(f"Input tokens: {input_tokens}")

            # Stream the response with Ollama
            stream = await client.chat(
                model="gemma3:12b",  # Adjust if the exact tag differs
                messages=messages,
                options={
                    "max_tokens": 1500,
                    "temperature": 0.7,
                },
                stream=True,
            )

            accumulated_response = ""
            async for chunk in stream:
                delta_content = chunk['message']['content']  # Ollama's streaming response format
                accumulated_response += delta_content
                await websocket.send_text(delta_content)

            # Calculate output tokens
            output_tokens = count_tokens(accumulated_response)
            print(f"Output tokens: {output_tokens}")

            # Send token counts to the client
            token_info = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            await websocket.send_text(f"\nToken usage: {json.dumps(token_info)}")

            print(f"Response: {accumulated_response}")
            print(f"Token usage: {token_info}")
    except WebSocketDisconnect:
        print("User disconnected.")
        await websocket.send_text("Goodbye!")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_text("Error occurred. Please try again.")
    finally:
        print("Closing WebSocket connection.")
        await websocket.close()