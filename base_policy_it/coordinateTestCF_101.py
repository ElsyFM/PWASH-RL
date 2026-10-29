
#!/usr/bin/env python3

import time
import signal
import sys
import argparse
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig


# ============================================================
# CONFIG
# ============================================================

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"

_running = True
_cf = None


# ============================================================
# SAFETY
# ============================================================

def signal_handler(sig, frame):
    global _running

    print("\n\n!!! EMERGENCY STOP !!!")

    _running = False

    try:
        if _cf:
            _cf.commander.send_stop_setpoint()
            time.sleep(0.2)
            _cf.close_link()
    except Exception:
        pass

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# MAIN
# ============================================================

def main(args):
    global _cf

    print("Initializing Crazyflie drivers...")
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache="./cache")
    _cf = cf

    connected_event = threading.Event()

    def on_connected(uri):
        print(f"\nConnected to {uri}")
        connected_event.set()

    def on_disconnected(uri):
        print("\nDisconnected.")

    cf.connected.add_callback(on_connected)
    cf.disconnected.add_callback(on_disconnected)

    print(f"\nConnecting to {args.uri}...")
    cf.open_link(args.uri)

    # --------------------------------------------------------
    # Wait for connection
    # --------------------------------------------------------

    if not connected_event.wait(timeout=10):
        print("ERROR: Connection timeout.")
        return

    print("\nWaiting for estimator position...")

    # --------------------------------------------------------
    # Position variables
    # --------------------------------------------------------

    current_position = {
        "x": None,
        "y": None,
        "z": None
    }

    # --------------------------------------------------------
    # Logging callback
    # --------------------------------------------------------

    def log_data(timestamp, data, logconf):

        if "stateEstimate.x" in data:
            current_position["x"] = data["stateEstimate.x"]

        if "stateEstimate.y" in data:
            current_position["y"] = data["stateEstimate.y"]

        if "stateEstimate.z" in data:
            current_position["z"] = data["stateEstimate.z"]

    log_config = None

    try:

        # ====================================================
        # CREATE POSITION LOG
        # ====================================================

        log_config = LogConfig(
            name="Position",
            period_in_ms=100
        )

        log_config.add_variable("stateEstimate.x", "float")
        log_config.add_variable("stateEstimate.y", "float")
        log_config.add_variable("stateEstimate.z", "float")

        log_config.data_received_cb.add_callback(log_data)

        # IMPORTANT:
        # Register the log configuration with the Crazyflie
        cf.log.add_config(log_config)

        if not log_config.valid:
            print("\nERROR: Position log configuration is invalid.")
            print("The firmware may not expose stateEstimate.x/y/z.")
            return

        log_config.start()

        # Give the estimator time to provide data
        time.sleep(2)

        # ====================================================
        # CURRENT POSITION
        # ====================================================

        print("\n========================================")
        print(" CURRENT ESTIMATED POSITION")
        print("========================================")

        if current_position["x"] is None:
            print("ERROR: No estimator data received.")
            print("Check the Loco positioning / estimator.")
            return

        print(f"X: {current_position['x']:.3f} m")
        print(f"Y: {current_position['y']:.3f} m")
        print(f"Z: {current_position['z']:.3f} m")

        print("========================================")

        # ====================================================
        # PRINT ONLY
        # ====================================================

        if args.print_only:

            print("\nPRINT-ONLY MODE")
            print("No flight commands were sent.")

            # Keep displaying position for a few seconds
            if args.monitor:
                print("\nMonitoring estimator. Press Ctrl+C to stop.")

                while _running:

                    if current_position["x"] is not None:
                        print(
                            f"X={current_position['x']: .3f}  "
                            f"Y={current_position['y']: .3f}  "
                            f"Z={current_position['z']: .3f}"
                        )

                    time.sleep(0.5)

            return

        # ====================================================
        # TARGET
        # ====================================================

        print("\n========================================")
        print(" TARGET POSITION")
        print("========================================")

        print(f"X:   {args.x:.3f} m")
        print(f"Y:   {args.y:.3f} m")
        print(f"Z:   {args.z:.3f} m")
        print(f"Yaw: {args.yaw:.1f}°")

        if args.duration is None:
            print("Duration: INDEFINITE")
            print("Stop with Ctrl+C")
        else:
            print(f"Duration: {args.duration:.1f} seconds")

        print("========================================")

        # ====================================================
        # CONFIRMATION
        # ====================================================

        print("\nThe Crazyflie WILL receive position commands.")

        input(
            "Press ENTER to begin the flight test, "
            "or Ctrl+C to cancel..."
        )

        # ====================================================
        # FLIGHT TEST
        # ====================================================

        print("\n========================================")
        print(" POSITION COMMAND ACTIVE")
        print("========================================")

        print(
            f"TARGET: "
            f"X={args.x:.3f} "
            f"Y={args.y:.3f} "
            f"Z={args.z:.3f}"
        )

        print("\nLive estimator data:")
        print("Press Ctrl+C to STOP.")
        print("========================================")

        start_time = time.time()

        last_print = 0

        while _running:

            # ------------------------------------------------
            # SEND POSITION SETPOINT
            # ------------------------------------------------

            cf.commander.send_position_setpoint(
                args.x,
                args.y,
                args.z,
                args.yaw
            )

            # ------------------------------------------------
            # PRINT LIVE POSITION
            # ------------------------------------------------

            now = time.time()

            # Print every 0.5 seconds
            if now - last_print >= 0.5:

                last_print = now

                x = current_position["x"]
                y = current_position["y"]
                z = current_position["z"]

                if x is not None and y is not None and z is not None:

                    error_x = args.x - x
                    error_y = args.y - y
                    error_z = args.z - z

                    distance = (
                        error_x ** 2
                        + error_y ** 2
                        + error_z ** 2
                    ) ** 0.5

                    print(
                        f"POS: "
                        f"X={x: .3f} "
                        f"Y={y: .3f} "
                        f"Z={z: .3f}"
                        f" | "
                        f"ERR: "
                        f"X={error_x:+.3f} "
                        f"Y={error_y:+.3f} "
                        f"Z={error_z:+.3f}"
                        f" | "
                        f"DIST={distance:.3f}m"
                    )

                else:
                    print("Waiting for estimator data...")

            # ------------------------------------------------
            # OPTIONAL DURATION
            # ------------------------------------------------

            if (
                args.duration is not None
                and now - start_time >= args.duration
            ):
                print("\nRequested duration reached.")
                break

            # Send setpoint at approximately 10 Hz
            time.sleep(0.1)

        # ====================================================
        # STOP
        # ====================================================

        print("\nSending STOP setpoint...")

        cf.commander.send_stop_setpoint()

        time.sleep(0.5)

        print("Flight test finished.")

    finally:

        try:
            if log_config:
                log_config.stop()
        except Exception:
            pass

        try:
            cf.commander.send_stop_setpoint()
        except Exception:
            pass

        try:
            cf.close_link()
        except Exception:
            pass


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Crazyflie Loco Positioning coordinate test"
    )

    parser.add_argument(
        "--uri",
        default=DEFAULT_URI,
        help="Crazyflie radio URI"
    )

    parser.add_argument(
        "--x",
        type=float,
        default=0.0,
        help="Target X coordinate in meters"
    )

    parser.add_argument(
        "--y",
        type=float,
        default=0.0,
        help="Target Y coordinate in meters"
    )

    parser.add_argument(
        "--z",
        type=float,
        default=0.3,
        help="Target Z coordinate in meters"
    )

    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Target yaw in degrees"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Flight duration in seconds. "
             "If omitted, run until Ctrl+C."
    )

    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print current position without sending flight commands."
    )

    parser.add_argument(
        "--monitor",
        action="store_true",
        help="With --print-only, continuously print position."
    )

    args = parser.parse_args()

    main(args)
