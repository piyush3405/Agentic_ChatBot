from agent import get_agent
from langchain_core.messages import HumanMessage
from database import init_db

init_db()  # Ensure the database is initialized before running the main function

def main():
    try:
        print("1. Creating chatbot agent with fallback chain...", flush=True)

        # You can also pass a specific model here if you want, e.g., get_agent("gemini-3.5-flash-lite")
        agent = get_agent()

        print("2. Agent created successfully", flush=True)

        config = {
            "configurable": {
                "thread_id": "test_thread_id",
            },
            "recursion_limit": 10,
        }

        print("3. Starting chatbot execution...", flush=True)

        # Stream the response; if primary model hits 429/503, fallbacks execute automatically
        for message_chunk, metadata in agent.stream(
            {
                "messages": [
                    HumanMessage(content="remember my name is Piyush"),
                ]
            },
            config=config,
            stream_mode="messages",
        ):
            node_name = metadata.get("langgraph_node", "Unknown")

            if message_chunk.content:
                print(
                    f"\n[{node_name}] ",
                    end="",
                    flush=True,
                )
                print(
                    message_chunk.content,
                    end="",
                    flush=True,
                )

        print("\n\n4. Execution completed", flush=True)

    except KeyboardInterrupt:
        print("\nExecution stopped by user (Ctrl+C).", flush=True)

    except Exception as e:
        print(
            f"\nExecution failed: {type(e).__name__}: {e}",
            flush=True,
        )

    finally:
        print("5. Main function cleanup reached.", flush=True)


if __name__ == "__main__":
    main()