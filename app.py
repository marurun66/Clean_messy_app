import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def main():
    st.markdown("<h1>방이 깨끗한지 🧼✨<br>더러운지 🗑️🤢 확인해보세요!</h1>", unsafe_allow_html=True)
    st.info('🧞‍♂️방 사진을 올려주시면, 방이 깨끗한지 더러운지 확인해드립니다!')
    image = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png','webp'])
    if image is not None:
        #유저가 올린 사진을 화면에 표시
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        image=Image.open(image)
        # 스트림릿(st.file_uploader())을 통해 업로드된 이미지 데이터를 
        # PIL(Python Imaging Library)에서 읽을 수 있는 형식으로 변환하기 위해서

        #모델 로드
        model = load_model("model/keras_model.h5", compile=False)
        print(model)

        # Load the labels
        class_names = open("model/labels.txt", "r").readlines()

        # 비어있는 넘파이 데이터를 하나 만듬
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # 이미 이미지는 불러왔음 이 코드는 필요없다.
        # image = Image.open("<IMAGE_PATH>").convert("RGB")

        # 이미지를 224x224로 resize
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        # 이미지 파일을 배열로 변환 (인공지능이 예측하려면 넘파이 배열이 필요하다)
        image_array = np.asarray(image)

        # 정규화 (0~1사이의 값으로 변환)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # 데이터에 0번째 인덱스에 이미지 배열을 넣음
        data[0] = normalized_image_array
        
        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        # 0은 클린, 1은 메시
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        # 클래스 네임의 두번째 인덱스부터 끝까지 출력
        # 0 clean, 1 mess
        # c부터 끝까지, m부터 끝까지 즉 clean, mess만 출력
        print("Confidence Score:", confidence_score)
        st.info(f'🧞‍♂️이 방은 {class_name[2:]}방 입니다. 제가 맞출 확률은 {confidence_score*100:.2f}% 입니다.')


        


if __name__ == '__main__':
    main()
    