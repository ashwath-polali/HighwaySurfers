import os
import traceback

from bot.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        # Startup/teardown failures (window not found, capture init, etc.) land
        # here. Write the full traceback next to the project so it's readable
        # even when the console has already scrolled away.
        tb = traceback.format_exc()
        print(tb)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write("=== startup/teardown crash ===\n" + tb + "\n")
        except OSError:
            pass
        print(f"\nWrote details to {path}")
