import os
import openai
import cv2
import base64
from dotenv import load_dotenv
# import img_segmentation
import get_streetview
from record_time import TimeRecord

load_dotenv(verbose=True)

# openai.api_key = os.environ["OPENAI_APIKEY"]
client = openai.OpenAI(api_key=os.environ["OPENAI_APIKEY"])



def load_and_encode_images(image_sources):
    encoded_images = []
    for source in image_sources:
        encoded_images.append(base64.b64encode(cv2.imencode('.jpg', source)[1]).decode('utf-8'))
    return encoded_images

"""
당신은 주어진 이미지에서 도로의 차선의 갯수를 세려주는 도우미입니다. 
오직 도로의 차선 혹은 차량의 방향만 확인하고, 다른 것은 신경쓰지마세요.
각 이미지마다 아래의 케이스에 맞게 차선의 갯수를 세려주세요.
- 모든 차선을 세지 마세요. 이미지에서 바라보는 방향을 정방향으로 합니다. 정방향의 도로만 집중하세요.
- 차선이 제대로 보이지 않아도 차량의 위치를 보고 차선의 갯수를 유추해주세요.
- 차량의 앞모습은 역방향에서 오는 것이고, 뒷모습은 정방향으로 가는 방향입니다. 차량의 뒷모습이 있는 위치의 차선만 집중해주세요.
- 주차는 생각하지말고 오로지 차선에만 집중해주세요.
- 차선라인을 세지 마세요. 오로지 차선만 세주세요.

대답을 할 때는 반드시 아래의 예시에 맞게 대답해주세요.
이유는 상세히 설명해주세요.
'
1번째 이미지: 2
2번째 이미지: 3
3번째 이미지: 1
' 
"""
"""
The image is taken on a driving road. Please explain in detail the current road shape here.
"""
"""
                You are a helper who counts the number of lanes on the road in the given image.\n
                Only check the line of the road or the direction of the vehicle, never mind anything else.\n
                For each image, please count the number of lanes according to the case below.\n
                - Don't count all the lanes. Based on the direction you see in the image. Focus only on the roads in the forward direction.\n
                - Even if you can't see the lane properly, look at the location of the vehicle and guess the number of lanes.\n
                - The front view of the vehicle is from the opposite direction, and the back view is from the forward direction. Just focus on the lane where the back of the vehicle is located.\n
                - Don't think about parking, just focus on the lane.\n
                - Don't count the lines, just count lanes.\n
                
                When you answer, be sure to answer according to the example below.
                Please explain the reason in detail
                - case1:
                '
                1st image: 2 , reason: ...
                2nd image: 3 , reason: ...
                3rd image: 1 , reason: ...
                ...
                '
                - case2:
                '
                1st image: 3 , reason: ...
                '
                - case3:
                '
                1st image: 2 , reason: ...
                2nd image: The image does not show a road with visible lanes, so I cannot provide a count.
                ...
                '
            """
def process_image_detect(image_sources, prompt):
    base64_images = load_and_encode_images(image_sources)

    messages = [
        {
            "role": "system",
            "content": """
                You are a helper who counts the number of lanes on the road in the given image.\n
                Only check the line of the road or the direction of the vehicle, never mind anything else.\n
                For each image, please count the number of lanes according to the case below.\n
                - Don't count all the lanes. Based on the direction you see in the image. Focus only on the roads in the forward direction.\n
                - Even if you can't see the lane properly, look at the location of the vehicle and guess the number of lanes.\n
                - The front view of the vehicle is from the opposite direction, and the back view is from the forward direction. Just focus on the lane where the back of the vehicle is located.\n
                - Don't think about parking, just focus on the lane.\n
                - Don't count the lines, just count lanes.\n
                
                When you answer, be sure to answer according to the example below.
                Please explain the reason in detail
                - case1:
                '
                1st image: 2 , reason: ...
                2nd image: 3 , reason: ...
                3rd image: 1 , reason: ...
                ...
                '
                - case2:
                '
                1st image: 3 , reason: ...
                '
                - case3:
                '
                1st image: 2 , reason: ...
                2nd image: The image does not show a road with visible lanes, so I cannot provide a count.
                ...
                '
            """
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}} for base64_image in base64_images]
        }
    ]

    response = client.chat.completions.create(
        model = "gpt-4-vision-preview",
        messages = messages,
        max_tokens=1000,
        # response_format={"type": "json_object"}, # vision model에서는 json_object를 사용할 수 없음
        temperature=0.2
    )

    response_text = response.choices[0].message.content

    return response_text
"""
LINESTRING (127.11612688701085, 37.33810708731315, (하늘에서 찍힘)
127.11612690393095, 37.33781539884557, 127.1161643436555, 37.33749048049456, 
127.11620190882219, 37.33734052157168, 127.11626436636642, 37.337157201945885, 
127.11631432054513, 37.33705721995517, 127.1164143553941, 37.3368989003576, 
127.11655179849554, 37.33673222888593, 127.1167143444431, 37.33655725568272)
"""
"""
LINESTRING (127.15025996093222 37.46985873866847, 127.14933508109134 37.46986708442255, 127.1485226466036 37.469892072196465, 127.14812268608684 37.46990872890739)
"""
if __name__ == "__main__":
    record = TimeRecord()
    record.start()
    test_line = [
        [127.15025996093222, 37.46985873866847],
        [127.14933508109134, 37.46986708442255],
        [127.1485226466036, 37.469892072196465],
        [127.14812268608684, 37.46990872890739]
    ]

    # image_paths = ["./data/2215386.jpg","./data/2219233.jpg","./data/6677293.jpg","./data/6749031.jpg"]
    # image_paths = ["./data/3.jpg","./data/12.jpg","./data/14.jpg","./data/23.jpg","./data/81.jpg", "./data/90.jpg"]
    origin_img, filtered_imgs = get_streetview.run(test_line)
        

    prompt = "How many lanes are there in the current image?"
    # prompt = "Tell me about this image"

    response_text = process_image_detect(filtered_imgs, prompt)
    print(response_text)
    record.end(is_show=True)

    for i,image in enumerate(zip(origin_img,filtered_imgs)):
        cv2.imshow(f'Filtered_{i}', image[1])
        # cv2.imshow(f'Original_{i}', image[0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

