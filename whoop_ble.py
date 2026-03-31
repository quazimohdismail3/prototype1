import asyncio
from bleak import BleakClient

ADDRESS = "D5:23:27:F1:7F:F7"

async def connect_and_list_services():
    print("Connecting...\n")

    async with BleakClient(ADDRESS) as client:
        print("Connected:", client.is_connected)

        print("\nServices and Characteristics:\n")
        for service in client.services:
            print(f"[Service] {service.uuid}")
            for char in service.characteristics:
                print(f"  └── [Characteristic] {char.uuid}")

asyncio.run(connect_and_list_services())