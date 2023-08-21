import os
import cv2
import dlib
import numpy as np
from PIL import Image
import time
import shutil
import stat


def move_images(source_path, destination_folder, file_name1, file_name2):
    # 检查源路径是否存在
    if not os.path.exists(source_path):
        print("源路径不存在")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 构建源文件路径和目标文件路径
    source_file = source_path + file_name1
    destination_file = destination_folder + file_name2

    # 移动文件
    try:
        shutil.move(source_file, destination_file)
    except Exception as e:
        print(" ")


def rename(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 使用os.listdir()函数获取文件夹中的文件名列表
    file_names = os.listdir(folder_path)

    count = 1
    # 遍历文件名列表并修改每个文件名
    for file_name in file_names:
        old_filename = folder_path + "/" + file_name
        new_filename = folder_path + "/face_(1)" + str(count) + ".jpg"
        os.rename(old_filename, new_filename)
        count += 1
    file_names = os.listdir(folder_path)
    count = 1
    for file_name in file_names:
        old_filename = folder_path + "/" + file_name
        new_filename = folder_path + "/face_" + str(count) + ".jpg"
        os.rename(old_filename, new_filename)
        count += 1
    count -= 1
    return count


def extract_frames(video_path, output_path, frame_interval):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的帧率和总帧数
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("帧率:", fps)
    print("总帧数:", total_frames)

    # 设置每隔几帧抽取一帧
    interval = int(fps) * frame_interval

    # 初始化计数器
    count = 0

    while True:
        # 读取帧
        ret, frame = video.read()

        # 如果没有读取到帧或已经抽取到了指定数量的帧，则退出循环
        if not ret or count >= total_frames:
            break

        # 抽取帧并保存
        if count % interval == 0:
            output_frame_path = output_path + "/frame_" + str(count) + ".jpg"
            cv2.imwrite(output_frame_path, frame)
            # print("保存帧:", output_frame_path)

        count += 1

    # 释放视频文件
    video.release()


def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    # 遍历文件夹中的所有文件和文件夹
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):  # 如果是文件
            # 删除文件
            os.remove(file_path)

        elif os.path.isdir(file_path):  # 如果是文件夹
            # 递归清空文件夹
            clear_folder(file_path)
            # 删除空文件夹
            os.rmdir(file_path)


def has_face(save_path, file_path, detector, predictor5, k):
    for i in range(1, k):
        image_path = file_path + "/face_" + str(i) + ".jpg"
        image_path1 = save_path + "/face_" + str(i) + ".jpg"
        # 加载图像
        image = dlib.load_rgb_image(image_path)

        # 使用人脸检测器检测人脸
        faces = detector(image)

        # 判断是否检测到人脸
        if len(faces) > 0:
            # 获取第一个人脸的关键点
            shape = predictor5(image, faces[0])

            # 判断关键点数量是否满足要求
            if len(shape.parts()) < 5:  # 使用五个关键点进行判断
                # 设置需要修改的权限（给予删除权限）
                new_permissions = stat.S_IWRITE
                # 修改文件权限
                os.chmod(image_path, new_permissions)
                os.chmod(image_path1, new_permissions)
                # 删除文件
                os.remove(image_path)
                os.remove(image_path1)
        if len(faces) <= 0:
            # 设置需要修改的权限（给予删除权限）
            new_permissions = stat.S_IWRITE
            # 修改文件权限
            os.chmod(image_path, new_permissions)
            os.chmod(image_path1, new_permissions)
            # 删除文件
            os.remove(image_path)
            os.remove(image_path1)


def reshape(image_path0, count, save_path, save_path_rgb):
    global start_time3, start_time2, start_time1
    k = 1
    for i in range(1, count + 1):
        image_path = image_path0 + "/face_"
        img = image_path + str(i) + ".jpg"
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) < 1:
            continue
        for (x, y, w, h) in faces:
            # 裁剪人脸区域并调整大小为150x150像素
            face_chip = cv2.resize(image[y:y + h, x:x + w], (150, 150))
            image_rgb = cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB)
            # 保存待处理图片和用于观看的图片
            cv2.imwrite(save_path + "/face_" + str(k) + ".jpg", image_rgb)
            cv2.imwrite(save_path_rgb + "/face_" + str(k) + ".jpg", face_chip)
            k += 1
    return k


def extract_face_landmarks(image, predictor68, detector):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测图像中的人脸
    faces = detector(gray)

    # 保存特征点
    landmarks = []

    # 提取特征点
    for face in faces:
        # 使用关键点检测器检测人脸关键点
        face_landmarks = predictor68(gray, face)

        # 将关键点坐标保存到列表中
        for i in range(68):
            x = face_landmarks.part(i).x
            y = face_landmarks.part(i).y
            landmarks.append((x, y))

    return landmarks


def compare_image_quality(landmarks, image2, predictor68, detector, img2, img2_1):
    print(" ")
    # 提取图像2的特征点
    landmarks2 = extract_face_landmarks(image2, predictor68, detector)
    ii = []
    i = 1
    k = 0
    if len(landmarks2) == 0:
        # 设置需要修改的权限（给予删除权限）
        new_permissions = stat.S_IWRITE
        # 修改文件权限
        os.chmod(img2, new_permissions)
        os.chmod(img2_1, new_permissions)
        # 删除文件
        os.remove(img2)
        os.remove(img2_1)
        ii.append(k)
        return ii
    for landmark in landmarks:
        # 比较特征点的差异
        diff = 0
        for (x1, y1), (x2, y2) in zip(landmark, landmarks2):
            diff += (x1 - x2) ** 2 + (y1 - y2) ** 2
        # 计算平均差异值
        avg_diff = diff / len(landmarks2)
        if avg_diff <= 180:
            k = i
            print(avg_diff)
            ii.append(k)
        i += 1
    if k == 0:
        # 设置需要修改的权限（给予删除权限）
        new_permissions = stat.S_IWRITE
        # 修改文件权限
        os.chmod(img2, new_permissions)
        os.chmod(img2_1, new_permissions)
        # 删除文件
        os.remove(img2)
        os.remove(img2_1)
    return ii


def retect(count, save_path, landmarks, predictor68, detector):
    arr = []
    for i in range(1, count + 1):
        img2 = save_path + "_rgb/face_" + str(i) + ".jpg"
        img2_1 = save_path + "/face_" + str(i) + ".jpg"
        image2 = np.array(Image.open(img2))
        l = compare_image_quality(landmarks, image2, predictor68, detector, img2, img2_1)
        for k in l:
            if k != 0:
                arr.append([i, k])
    return arr


def distan(face_chip1, face_chip2):
    face_descriptor1 = facerec.compute_face_descriptor(face_chip1)
    face_descriptor2 = facerec.compute_face_descriptor(face_chip2)
    # 计算欧氏距离
    desc1 = np.array(face_descriptor1)
    desc2 = np.array(face_descriptor2)
    euclidean_distance = np.linalg.norm(desc1 - desc2)
    print("欧氏距离 ", euclidean_distance)
    # 计算余弦相似度
    # cosine_similarity = np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
    # print("余弦相似度", cosine_similarity)
    # if cosine_similarity > 0.8 and euclidean_distance < 0.5:
    if euclidean_distance < 0.48:
        return True
    else:
        return False


def face_detect(save_path, arr, find_path):
    while len(arr) != 0:
        [i, k] = arr[0]
        img1 = find_path + "/face_" + str(k) + ".jpg"
        face_chip1 = cv2.imread(img1)
        img2 = save_path + "_rgb/face_" + str(i) + ".jpg"
        img2_1 = save_path + "/face_" + str(i) + ".jpg"
        face_chip2 = cv2.imread(img2)
        same_person = distan(face_chip1, face_chip2)
        if same_person:
            destination_folder_rgb = "D:/WorkSpace/video/src/same" + str(k)
            count = rename(destination_folder_rgb)
            file_name1 = "/face_" + str(i) + ".jpg"
            file_name2 = "/face_" + str(count+1) + ".jpg"
            source_path_rgb = save_path + "_rgb"
            arr = [x for x in arr if not (x[0] == i)]
            print(arr)
            try:
                move_images(source_path_rgb, destination_folder_rgb, file_name1, file_name2)
            except FileNotFoundError:
                print(" ")
        else:
            arr.remove([i, k])
            ar = [x for x in arr if (x[0] == i)]
            if len(ar) == 0:
                # 设置需要修改的权限（给予删除权限）
                new_permissions = stat.S_IWRITE
                # 修改文件权限
                os.chmod(img2, new_permissions)
                os.chmod(img2_1, new_permissions)
                # 删除文件
                os.remove(img2)
                os.remove(img2_1)


# 开始计时
start_time0 = time.time()
num = 2
# 加载视频文件夹
video_path = "C:/Users/1/Desktop/my project/video/" + str(num) + ".mp4"
# 加载图片文件夹
image_path0 = "C:/Users/1/Desktop/my project/photo" + str(num)
frame_interval = 2  # 每隔frame_interval秒抽取一帧
clear_folder(image_path0)
extract_frames(video_path, image_path0, frame_interval)
count = rename(image_path0)
# 开始计时
start_time1 = time.time()
# 加载处理器
predictor68 = dlib.shape_predictor('D:/WorkPython/pythonProject/test2/shape_predictor_68_face_landmarks.dat')
predictor5 = dlib.shape_predictor('D:/WorkPython/pythonProject/test2/shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('D:/WorkPython/pythonProject/test2/dlib_face_recognition_resnet_model_v1.dat')
face_cascade = cv2.CascadeClassifier('D:/WorkPython/pythonProject/test2/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
# 预处理文件夹名称
save_path_rgb = image_path0 + "_Face_rgb"
clear_folder(save_path_rgb)
save_path = image_path0 + "_Face"
clear_folder(save_path)

# 处理图片（耗时最长）
k = reshape(image_path0, count, save_path, save_path_rgb)
# 检测是否出现人脸
has_face(save_path, save_path_rgb, detector, predictor5, k)
# 再次检测，删除误检测；检测可能是同一个人脸的照片
find_path = "D:/WorkSpace/video/src/find"
count = rename(find_path)
landmarks = []
for i in range(1, count + 1):
    img = find_path + "/face_" + str(i) + ".jpg"
    image = np.array(Image.open(img))
    landmark = extract_face_landmarks(image, predictor68, detector)
    landmarks.append(landmark)
# 整理文件夹
rename(save_path)
count = rename(save_path_rgb)
arr = retect(count, save_path, landmarks, predictor68, detector)
print(arr)

# 检测是否同一个人（一张图1秒左右）
face_detect(save_path, arr, find_path)
# 结束计时
start_time2 = time.time()
execution_time1 = start_time2 - start_time1
print("程序运行时间1：", execution_time1, "秒")

'''start_time1 = time.time()
start_time2 = time.time()
execution_time1 = start_time2 - start_time1
print("程序运行时间1：", execution_time1, "秒")
start_time3 = time.time()
execution_time2 = start_time3 - start_time2
print("程序运行时间2：", execution_time2, "秒")'''
