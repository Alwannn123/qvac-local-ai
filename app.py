import asyncio, sys
from tetherto.qvac_sdk import Client, completion, load_model, unload_model
from tetherto.qvac_sdk.models import LLAMA_3_2_1B_INST_Q4_0

async def main():
    print("QVAC Local AI - 100% on-device, no API key")
    async with Client() as client:
        t = client.transport
        print("Loading model...")
        model_id = await load_model(t, model_src=LLAMA_3_2_1B_INST_Q4_0)
        print("Model loaded! Type exit to quit")
        while True:
            q = input("You: ")
            if q.lower() == "exit": break
            run = completion(t, model_id=model_id, history=[{"role":"user","content":q}])
            print("AI: ", end="")
            async for e in run.events:
                if e.type == "contentDelta":
                    sys.stdout.write(e.text)
                    sys.stdout.flush()
            print()
        await unload_model(t, model_id)

asyncio.run(main())
