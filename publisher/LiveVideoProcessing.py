import argparse
import asyncio
import logging
import os
from pathlib import Path


from vehicle import set_torque


import cv2
import numpy as np
from kuksa_client.grpc import DataEntry
from kuksa_client.grpc import Datapoint
from kuksa_client.grpc import EntryUpdate
from kuksa_client.grpc import Field
from kuksa_client.grpc import VSSClientError
from kuksa_client.grpc.aio import VSSClient

import sys
import gi
gi.require_version('GLib', '2.0')
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import cv2
import time
import CameraData
import globals as G

Gst.init(None)
G.IP_ADDRESS="0.0.0.0"
G.PORT=9000

def get_angle_of_line(l, alpha):
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]

    angle = 0
    try:
        # angle = np.rad2deg(np.arctan((y2-y1)/(x2-x1)))
        angle = np.rad2deg(np.arctan((x2-x1)/(y2-y1)))
    except ZeroDivisionError:
        pass
    return angle

def get_offset(l, width_img, alpha):
    # get distance of endpoint closest to center vertical
    center = width_img//2
    x1 = l[0]
    y1 = l[1]
    x2 = l[2]
    y2 = l[3]

    if abs(x1-center) < abs(x2-center):
        return alpha*(x1-center)
    else:
        return alpha*(x2-center)


def determine_stearing(angle, offset):
    if abs(angle) > 35:
        sign = 1 if angle > 0 else -1
        angle = sign*35

    return angle/35


def indicator(frame):
    height, width, channel = frame.shape

    x_mid = int(width / 2)
    y_upper_mid = 0
    y_down_mid = height

    alpha = 3 # Contrast control (1.0-3.0)
    beta = 5 # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    ret, thresh1 = cv2.threshold(frame, 230, 255, cv2.THRESH_BINARY) 

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

    #cv2.line(closing, (x1Mean, y1Mean), (x2Mean, y2Mean), (255,0,0), 3)
    #cv2.line(closing, (x_mid, y_upper_mid), (x_mid, height), (0,255,0), 3)

    kernel = np.ones((5,5),np.float32)/25
    closing = cv2.filter2D(closing,-1,kernel)

    ret, thresh = cv2.threshold(closing, 100, 255, cv2.THRESH_BINARY)


    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(closing, 2, np.pi/180, 50,minLineLength=50, maxLineGap=100)

    x1Mean = []
    x2Mean = []

    y1Mean = []
    y2Mean = []

    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x1Mean.append(x1)
        x2Mean.append(x2)
        y1Mean.append(y1)
        y2Mean.append(y2)

    x1Mean = int(np.floor(np.mean(x1Mean)))
    x2Mean = int(np.floor(np.mean(x2Mean)))
    y1Mean = int(np.floor(np.mean(y1Mean)))
    y2Mean = int(np.floor(np.mean(y2Mean)))

    cv2.line(thresh, (x1Mean, y1Mean), (x2Mean, y2Mean), (255, 0, 0), 5)

    angle = get_angle_of_line([x1Mean,y1Mean,x2Mean,y2Mean], 100)

    print("Angle: {}°".format(int(np.floor(angle))))

    cv2.imshow("Frame", thresh)

def init_argparse() -> argparse.ArgumentParser:
    """This inits the argument parser for the CSV-provider."""
    parser = argparse.ArgumentParser(
        usage="-a [BROKER ADDRESS] -p [BROKER PORT] -f [FILE]",
        description="This provider writes the content of a csv file to a kuksa.val databroker",
    )
    environment = os.environ
    parser.add_argument(
        "-a",
        "--address",
        default=environment.get("KUKSA_DATA_BROKER_ADDR", "192.168.17.78"),
        help="This indicates the address of the kuksa.val databroker to connect to."
             " The default value is 127.0.0.1",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=environment.get("KUKSA_DATA_BROKER_PORT", "55555"),
        help="This indicates the port of the kuksa.val databroker to connect to."
             " The default value is 55555",
        type=int,
    )
    parser.add_argument(
        "-l",
        "--log",
        default=environment.get("PROVIDER_LOG_LEVEL", "INFO"),
        help="This sets the logging level. The default value is WARNING.",
        choices={"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"},
    )
    parser.add_argument(
        "--cacertificate",
        help="Specify the path to your CA.pem. If used provider will connect using TLS",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--tls-server-name",
        help="TLS server name, may be needed if addressing a server by IP-name",
        nargs="?",
        default=None,
    )
    return parser


async def main():
    """the main function as entry point for the CSV-provider"""
    parser = init_argparse()
    args = parser.parse_args()
    numeric_value = getattr(logging, args.log.upper(), None)
    if args.cacertificate:
        root_path = Path(args.cacertificate)
    else:
        root_path = None
    if isinstance(numeric_value, int):
        logging.basicConfig(encoding="utf-8", level=numeric_value)
    try:
        async with VSSClient(
                args.address,
                args.port,
                root_certificates=root_path,
                tls_server_name=args.tls_server_name,
        ) as client:
            await vehicle_control(client)
    except VSSClientError:
        logging.error(
            "Could not connect to the kuksa.val databroker at %s:%s."
            " Make sure to set the correct connection details using --address and --port"
            " and that the kuksa.val databroker is running.",
            args.address,
            args.port,
        )


async def pub_value(client, rows):
    for row in rows:
        if row["field"] == "current":
            # print(row['field'])
            entry = DataEntry(
                row["signal"],
                value=Datapoint(value=row["value"]),
            )
            updates = (EntryUpdate(entry, (Field.VALUE,)),)
            logging.info(
                "Update current value of %s to %s", row["signal"], row["value"]
            )
        elif row["field"] == "target":
            # print(row['field'])
            entry = DataEntry(
                row["signal"], actuator_target=Datapoint(value=row["value"])
            )
            updates = (EntryUpdate(entry, (Field.ACTUATOR_TARGET,)),)
            logging.info("Update target value of %s to %s", row["signal"], row["value"])
        else:
            updates = []
        try:
            await client.set(updates=updates)
        except VSSClientError as ex:
            logging.error("Error while updating %s\n%s", row["signal"], ex)
        try:
            await asyncio.sleep(delay=float(row["delay"]))
        except ValueError:
            logging.error(
                "Error while waiting for %s seconds after updating %s to %s."
                " Make sure to only use numbers for the delay value.",
                row["delay"],
                row["signal"],
                row["value"],
            )

def start_deceleration(torque):
    points = np.flip(np.linspace(0, torque, 10))[1:]
    for point in points:
        set_torque(point, 100)
    

async def vehicle_control(client):
    vehicle_camera=CameraData.GstUdpCamera(G.PORT)
    vehicle_camera.play()
    while True:
        if not vehicle_camera.new_imgAvaiable:
            print("no data")
            time.sleep(5)
            print("sleeping for 5 seconds")
        img = vehicle_camera.new_imgData
        print(img)
        frame=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        low_b = np.uint8([236, 240, 235])
        high_b = np.uint8([0, 0, 0])
        mask = cv2.inRange(frame, high_b, low_b)
        contours, hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)

        value = []
        should_drive_straight = await drive_straight(frame)
        detected_parking_lot = await detect_parking_sign(frame)
        indicator(frame)

        if not should_drive_straight or detected_parking_lot:
            print("drive straight: {},detected_parking_lot : {}".format(should_drive_straight, detected_parking_lot))
            value = [
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.IsEnabled",
                    "value": "FALSE",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Powertrain.Transmission.ClutchEngagement",
                    "value": "0",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Powertrain.Transmission.SelectedGear",
                    "value": "3",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.Brake",
                    "value": "1.0",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.SteeringAngle",
                    "value": "1",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.Torque",
                    "value": "0.0",
                    "delay": "0",
                },
            ]
        else:
            print("drive straight: {},detected_parking_lot : {}".format(should_drive_straight, detected_parking_lot))
            value = [
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.IsEnabled",
                    "value": "TRUE",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Powertrain.Transmission.ClutchEngagement",
                    "value": "0",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Powertrain.Transmission.SelectedGear",
                    "value": "3",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.Brake",
                    "value": "0",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.SteeringAngle",
                    "value": "0",
                    "delay": "0",
                },
                {
                    "field": "target",
                    "signal": "Vehicle.Teleoperation.Torque",
                    "value": "0.7",
                    "delay": "0",
                },
            ]

            #cv2.imshow("Frame", frame)
        await pub_value(client, value)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
            break
    cap.release()
    cv2.destroyAllWindows()


async def drive_straight(image):
    height, width = image.shape[:2]
    third_width = width // 3
    center_third = image[:, third_width:2 * third_width]
    height = center_third.shape[0]
    third_height = height // 3
    frame = center_third[2 * third_height:, :]
    ret, thresh1 = cv2.threshold(frame, 254, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(closing, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=100)
    closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    if lines is None:
        return False
    return True


async def detect_parking_sign(frame):
    height, width = frame.shape[:2]
    bottom_half = frame[(height // 2) - 50:, :]
    gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(1000))
    (corners, ids, rejected) = detector.detectMarkers(gray)
    if len(corners) > 0:
        print("found corner")
        if ids == 10:
            print("found 10")
            return True
    return False


asyncio.run(main())
