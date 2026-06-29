import os, time, shutil, cv2, sys
from datetime import datetime
import numpy as np
from queue import Queue
from threading import Thread
from collections import Counter

from ultralytics import YOLO
from PIL import Image
from glob import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from DBConnection import *
from Utils import *
from paths import *

# GLOBAL 
file_queue = Queue()
buffer = {}
rep_list = set()

# 실시간 추론용 이미지 경로
inf_inp = INFER_INP_IMG_DIR
inf_out = INFER_OUT_IMG_DIR

pred_dir = PREDICT_DIR

class Handler(FileSystemEventHandler):
    def __init__(self, line_id, file_queue):
        super().__init__()
        self.line_id = line_id
        self.file_queue = file_queue
        
    # 감시 폴더 내 파일 생성 이벤트 처리
    def on_created(self, event):
        # 디렉토리 생성 이벤트 제외
        if event.is_directory:
            print(f"[WATCHDOG][SKIP] Directory Created -> {event.src_path}")
            return
        
        created_time = datetime.now() # .strftime("%Y-%m-%d")
        Fname, Extension = os.path.splitext(os.path.basename(event.src_path))

         # 이미지 파일만 Queue 적재
        if Extension.lower() not in (".jpg", ".jpeg", ".png", ".bmp"): 
            print(f"[WATCHDOG][SKIP] Not Image -> {Fname}{Extension}")
            return
        else:
            self.file_queue.put({
                "created_time": created_time,
                "created_path": event.src_path,
                "line_id": self.line_id
            })
            print(f"[WATCHDOG][QUEUE] {self.line_id} -> {Fname}{Extension}")


def dispatcher():
    """
    Watchdog Queue 데이터를 받아 buffer에 저장하고
    Left / Center / Right 이미지가 모두 모이면 processor로 전달한다.
    """
    while True:
        # Watchdog가 Queue에 적재한 이미지 정보 수신
        item = file_queue.get()
        print(f"[DISPATCHER][GET] {item}")

        created_time = item["created_time"]
        created_path = item["created_path"]
        line_id = item["line_id"]
        Fname, Extension = os.path.splitext(os.path.basename(created_path))

        # 동일 파일 중복 처리
        if Fname in rep_list:
            print(f"[DISPATCHER][DUPLICATE] {Fname}")
            if f"{Fname}{Extension}" in os.listdir(SHARED):
                os.remove(f"{SHARED}\\{Fname}{Extension}")
                continue 
            else: continue
        else:
            # 중복 방지 목록 등록
            rep_list.add(Fname)
            
            # Line03/Line04 + 카메라 위치를 DB용 카메라 번호로 변환
            line_num = 3 if line_id == "Line03" else 4 if line_id == "Line04" else None
            cam_pos, cam_base = (
                ("Left", 1) if "left" in created_path.lower() else
                ("Center", 2) if "center" in created_path.lower() else
                ("Right", 3) if "right" in created_path.lower() else
                (None, None)
            )

            cam_idx = (line_num - 3) * 3 + cam_base
            cam_num = f"CAM{cam_idx:02d}" if line_num and cam_base else None

            # 파일명 기준(YYYY-MM-DD-HH-MM-SS) 그룹 Key 생성
            # 기존 저장 파일명들로 보아 대부분 동일한 시점의 촬영 이미지는 동일한 시각의 파일명으로 생성됨
            # 예: ('Line03', '2026-06-15-10-25-32')
            base_key = Fname.rsplit("_", 1)[0]
            key = (line_id, base_key)

            # 동일 시점의 Left/Center/Right 이미지 저장 버퍼 생성
            if key not in buffer:
                buffer[key] = {
                    "Left": None,
                    "Center": None,
                    "Right": None
                }

            # 카메라 위치별 이미지 정보 저장
            buffer[key][cam_pos] = {
                "created_time": created_time,
                "created_path": created_path,
                "line_id": line_id,
                "fname": Fname,
                "ext": Extension,
                "cam_pos": cam_pos,
                "cam_num": cam_num,
            }
            print(f"[DISPATCHER][BUFFER][PUT] key={key}")
          
def processor(model, classes):
    """
    buffer에 저장된 3개 카메라 이미지가 모두 수집되었을 때
    inference를 실행하고 process()로 전달한다.
    """
    while True:
        # buffer에 Left/Center/Right 3장 모두 모인 key만 처리
        for key, cams in list(buffer.items()):
            if all([cams["Left"], cams["Center"], cams["Right"]]):
                print(f"[PROCESSOR][READY] key={key}")
                items = [cams["Left"], cams["Center"], cams["Right"]]
                process(items, model, classes)
                del buffer[key]
        time.sleep(0.2)

def select_main_object(result, img_shape):
            h, w = img_shape[:2]
            center = np.array([w / 2, h / 2])

            best = None
            best_score = -1

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                area = (x2 - x1) * (y2 - y1)

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dist = np.linalg.norm(np.array([cx, cy]) - center)

                score = conf * area / (dist + 1e-6)

                if score > best_score:
                    best_score = score
                    best = (int(x1), int(y1), int(x2), int(y2), cls)

            return best

def process(items, model, classes):
    """
    3개 카메라 이미지에 대해 YOLO inference를 수행하고
    DB 저장 및 결과 이미지 생성까지 처리한다.
    """ 

    # 추론 이미지 세장 저장
    predict_imgs = {}
    # 카메라별 결과를 문자열로 합쳐 Monitoring 로그용으로 사용
    all_products = []

    # 최종 판정 제품
    detect_products = ""
    detect_products_cnt = []

    # 원본 이미지 정보 DB 저장
    for item in items:
        created_time = item["created_time"]
        img_path = item["created_path"]
        line_id = item["line_id"]
        Fname = item["fname"]
        Extension = item["ext"]
        cam_pos = item["cam_pos"]
        cam_num = item["cam_num"] 
        
        insert_data = {
            "LINEID": line_id,
            "CAM_NUM": cam_num,
            "DATIME": created_time,
            "DANAME": Fname + Extension,
            "FILEPATH": inf_inp / f"{line_id}" / f"{created_time.strftime('%Y-%m-%d')}",
            "REMARK": "",
        }
        try:
            # 원본 백업
            insert_data["FILEPATH"].mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, insert_data["FILEPATH"] / insert_data["DANAME"])
            insert("WM_DAVALUE_IMG", insert_data)
        except Exception as e:
            print(f"[PROCESSOR][ERROR] Save Original Image -> {e}")
            
        # 파일 저장 안정화 대기
        time.sleep(1)

        # YOLO 모델 추론 수행
        print(f"[YOLO][PREDICT] {Fname}")
        # model.predict(
        #     source=img_path, 
        #     save=True, 
        #     save_txt=True, 
        #     conf=0.65, 
        #     iou=0.4, 
        #     save_crop=True, 
        #     project=DETECT_DIR)

        "메인 Object 한개만 남김"
        results = model.predict(
            source=img_path,
            conf=0.65,
            iou=0.4,
            save=False
        )

        result = results[0]
        img = cv2.imread(img_path)
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        bbox = select_main_object(result, img.shape)
        if bbox is None:
            return None
        else:
            x1, y1, x2, y2, cls_id = bbox
            label = model.names[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            save_path = (
                Path(inf_out)
                / str(line_id)
                / created_time.strftime('%Y-%m-%d')
                / insert_data["DANAME"].replace(".jpg", "_boxed.jpg")
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img)

        all_products_txt = PREDICT_DIR / "labels" / f"{Fname}.txt"

        try:
            with open(all_products_txt, "r") as f:
                item_list = [classes[int(line.split()[0])] for line in f]
            counts=  Counter(item_list)
            detected_label = ", ".join(f"{name} - {cnt}" for name, cnt in counts.items())

            # Center 카메라 기준 최종 제품 수량 집계 (DB 저장용)
            if cam_pos == "Center": 
                detect_products = detected_label
                detect_products_cnt = [{"PRODUCTNAME": name, "PRODUCT_QTY": cnt} for name, cnt in counts.items()]

        except OSError:
            detected_label = '인식불가'
            # ================ Test Data ================
            import random
            product_names = [
                "A_PRODUCT",
                "B_PRODUCT",
                "C_PRODUCT",
                "D_PRODUCT"
            ]

            counts = [
                {"PRODUCTNAME": name,
                "PRODUCT_QTY": random.randint(1, 10)}
                for name in random.sample(product_names, random.randint(1, len(product_names)))
            ]
            detected_label = ", ".join(f"{detect_prod['PRODUCTNAME']} - {detect_prod['PRODUCT_QTY']}" for detect_prod in counts)
            if cam_pos == "Center": 
                detect_products = detected_label
                detect_products_cnt = counts
            # ===========================================
            print(f"[YOLO][PREDICT] {Fname} -> No Detection")
        

        # 카메라 위치별 제품명 문자열로 생성
        all_products.append(f"{cam_pos}: {detected_label}")
        
        # 추론 결과 이미지 저장
        predict_img = pred_dir / f"{Fname}{Extension}"
                
        if predict_img:
            img = cv2.imread(str(predict_img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            print(f"[YOLO][IMAGE] {Fname} -> Predict Image Not Found")
            org_img = cv2.imread(str(img_path))
            h, w = org_img.shape[:2]
            img = np.full((h, w, 3), 255, dtype=np.uint8)

        predict_imgs[cam_pos] = img
        
        # 임시 추론 결과 및 감시 폴더 이미지 정리
        shutil.rmtree(pred_dir, ignore_errors=True)
        

    # Left / Center / Right 순서로 이미지 병합하여 최종 결과 생성
    merged_img = np.hstack([predict_imgs["Left"], predict_imgs["Center"], predict_imgs["Right"]])
    merged_img_path = inf_out / f"{line_id}" / f"{created_time.strftime('%Y-%m-%d')}"
    merged_img_path.mkdir(parents=True, exist_ok=True)
    merged_img_name = f"{created_time.strftime('%Y-%m-%d-%H-%M-%S')}_merged{Extension}"
    Image.fromarray(merged_img).save(merged_img_path / merged_img_name)

    
    # 모니터링 결과 DB 저장
    insert_monitering_data = {
        "DATIME": created_time,
        "LINEID": line_id,
        "ALL_PRODUCTS": " | ".join(all_products),
        "DETECT_PRODUCTS": detect_products,
        "FILEPATH": merged_img_path,
        "FILENAME": merged_img_name,
        "ERROR_YN": "N",
        "CAM_NUM": cam_num,
        "REMARK": "",
    }
    
    try:
        insert("WM_MONITORING_IMG", insert_monitering_data)
    except Exception as e:
        print(f"[PROCESSOR][ERROR] Save Mointoring Data -> {e}")

    # 최종 판정 결과 제품별 생산량 DB 저장
    for product in detect_products_cnt:
        try:
            # 현재 수량 조회
            rows = select("WM_PRODUCT_CNT", columns="PRODUCT_QTY", where="DATIME=%s AND PRODUCTNAME=%s",
                        params=(created_time.strftime('%Y-%m-%d'), product["PRODUCTNAME"]))
            if rows:
                current_qty = int(rows[0][0])
                update(
                    "WM_PRODUCT_CNT",
                    {"PRODUCT_QTY": current_qty + product["PRODUCT_QTY"]},
                    "DATIME=%s AND PRODUCTNAME=%s",
                    (created_time.strftime('%Y-%m-%d'), product["PRODUCTNAME"])
                )
            else:
                insert_cnt_data = {
                    "DATIME": created_time.strftime('%Y-%m-%d'),
                    "PRODUCTNAME": product["PRODUCTNAME"],
                    "PRODUCT_QTY": product["PRODUCT_QTY"],
                    "REMARK": "",
                }
                insert("WM_PRODUCT_CNT", insert_cnt_data)
        except Exception as e:
            print(f"[PROCESSOR][ERROR] Save Count of Products -> {e}")
        
    
if __name__ == '__main__':
    print(f"[SYSTEM][START] BASE_DIR -> {BASE_DIR}")
    print(f"[SYSTEM][START] SHARED -> {SHARED}")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    INFER_INP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    INFER_OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 predict 폴더 삭제
    shutil.rmtree(PREDICT_DIR, ignore_errors=True)
    print("[SYSTEM][INIT] Predict Directory Removed")

    try: 
        result = select(table="WM_TRAIN_INFO", columns="MODELNAME", extra="ORDER BY TRAINDATE DESC LIMIT 1")
        modelname = result[0][0] if result else None
        if not modelname:
            raise ValueError("학습 모델 정보가 없습니다.")
        else: modelname = modelname.split('.')[0]

        model_path = DETECT_DIR / modelname / "weights" / "best.pt"
        final_model = YOLO(str(model_path))
        print(f"[SYSTEM][MODEL] Loaded -> {model_path}")

    except Exception as e:
        print(f"[SYSTEM][MODEL][ERROR] {e}")
        sys.exit(1)


    # 추론 결과 라벨 분석
    with open(CLASSES_TXT, 'r') as f:
        classes = f.read().splitlines()
    
    
    # THREAD START
    Thread(target=dispatcher, daemon=True).start()
    Thread(target=processor, args=(final_model, classes, ), daemon=True).start()
    print("[SYSTEM][THREAD] Dispatcher Started")
    print("[SYSTEM][THREAD] Processor Started")


    # WATCHDOG SETUP
    line03_dir = Path("D:/04_Image/LINE3") # 수집 이미지 저장 폴더 감시 디렉토리
    line04_dir = Path("D:/04_Image/LINE4")
    print("[SYSTEM][WATCHDOG] Monitoring Started")
    print(f"[SYSTEM][WATCHDOG] {line03_dir}")
    print(f"[SYSTEM][WATCHDOG] {line04_dir}")
    
    observer = Observer()
    observer.schedule(Handler("Line03", file_queue), str(line03_dir), recursive=True)
    observer.schedule(Handler("Line04", file_queue), str(line04_dir), recursive=True)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    except Exception as e:
        observer.stop()
        print(f"Error: {e}")
    finally:
        observer.join()
