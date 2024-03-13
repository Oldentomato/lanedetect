import os
from dotenv import load_dotenv
import requests
import math
import cv2
import io
import numpy as np
import img_processing

load_dotenv(verbose=True)

api_key = os.environ['GOOGLE_APIKEY']


def calc_data(coord_1, coord_2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(coord_1[1])
    lon1_rad = math.radians(coord_1[0])
    lat2_rad = math.radians(coord_2[1])
    lon2_rad = math.radians(coord_2[0])

    # Calculate differences
    d_lon = lon2_rad - lon1_rad

    # Calculate differences
    d_lon = lon2_rad - lon1_rad

    # Calculate bearing
    y = math.sin(d_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
    bearing = math.atan2(y, x)

    # Convert bearing from radians to degrees
    bearing_degrees = math.degrees(bearing)

    # Ensure bearing is within [0, 360) range
    bearing_degrees = (bearing_degrees + 360) % 360

    return bearing_degrees



def get_streetview_images(coord, dir):
    # Google Street View API 엔드포인트
    url = "https://maps.googleapis.com/maps/api/streetview"

    # API 요청 매개변수 설정
    params = {
        "size": "512x512",  # 이미지 크기 설정 (가로x세로)
        "location": f"{coord[1]},{coord[0]}",  # 시작점 좌표
        "heading": f"{dir}",  # 시작 방향 (0은 북쪽을 가리킴)
        "pitch": "0",  # 시야각 (0은 수평)
        "fov": "100",  # 시야각 설정 (값이 클수록 넓은 범위를 볼 수 있음)
        "key": api_key  # Google Maps API 키
    }

    # 시작점 로드뷰 이미지 가져오기
    response = requests.get(url, params=params)
    image = response.content


    return image

def run(linearr):
    image_arr = []
    origin_arr = []

    for i in range(0, len(linearr)-1):
        result_dir = calc_data(linearr[i], linearr[i+1])
        result = get_streetview_images(linearr[i], result_dir)
        result_image = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)

        # result_img = img_processing.image_histo(result_image)
        filtered_image = img_processing.image_bilateral(result_image)
        filtered_image = img_processing.img_contrast(filtered_image)
        # filtered_image = img_processing.image_clearing(filtered_image)
        image_arr.append(filtered_image)
        origin_arr.append(result_image)

        
    return origin_arr, image_arr

#debug
# if __name__ == "__main__":
#     run()