import os, time, shutil, sys
from datetime import datetime
from queue import Queue
from threading import Thread
from collections import Counter

# from ultralytics import YOLO
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from MultiViewModel import MultiviewYOLODetector
from DBConnection import *
from Utils import *
from config import *

# GLOBAL 
file_queue = Queue()
buffer = {}
rep_list = set()

# Image path for real-time inference
inf_inp = INFER_INP_IMG_DIR
inf_out = INFER_OUT_IMG_DIR

pred_dir = PREDICT_DIR

class Handler(FileSystemEventHandler):
    def __init__(self, monitor_dir, file_queue):
        super().__init__()
        self.monitor_dir = monitor_dir
        self.file_queue = file_queue

    # Handle file creation events
    def on_created(self, event):
        # Ignore directories
        if event.is_directory:
            print(f"[WATCHDOG][SKIP] Directory Created -> {event.src_path}")
            return
        
        created_time = datetime.now() # .strftime("%Y-%m-%d")
        Fname, Extension = os.path.splitext(os.path.basename(event.src_path))

        # Enqueue image files only
        if Extension.lower() not in (".jpg", ".jpeg", ".png", ".bmp"): 
            print(f"[WATCHDOG][SKIP] Not Image -> {Fname}{Extension}")
            return
        else:
            self.file_queue.put({
                "created_time": created_time,
                "created_path": event.src_path,
                "monitor_dir": self.monitor_dir
            })
            print(f"[WATCHDOG][QUEUE] {self.monitor_dir} -> {Fname}{Extension}")

def dispatcher():
    """
    Buffers image events received from the Watchdog queue.
    Forwards the complete image set to the processor
    when all camera images are available.
    """
    while True:
        # Get image event from Watchdog queue
        item = file_queue.get()
        print(f"[DISPATCHER][GET] {item}")

        created_time = item["created_time"]
        created_path = item["created_path"]
        monitor_dir = item["monitor_dir"]
        Fname, Extension = os.path.splitext(os.path.basename(created_path))

        # Ignore duplicate filenames
        if Fname in rep_list:
            print(f"[DISPATCHER][DUPLICATE] {Fname}")
            if f"{Fname}{Extension}" in os.listdir(SHARED):
                os.remove(f"{SHARED}\\{Fname}{Extension}")
                continue 
            else: continue
        else:
            # Register duplicate prevention list
            rep_list.add(Fname)
            
            # Convert line and camera position to DB camera number
            monitor_dir = "monitor_dir01" if monitor_dir == "monitor_dir01" \
                else "monitor_dir02" if monitor_dir == "monitor_dir02" else None
            cam_pos = (
                "Left" if "left" in created_path.lower() else
                "Center" if "center" in created_path.lower() else
                "Right" if "right" in created_path.lower() else
                None
            )

            # Group key based on timestamp (YYYY-MM-DD-HH-MM-SS)
            # Most images with the same shooting time share the same timestamp.
            base_key = Fname.rsplit("_", 1)[0]
            key = (monitor_dir, base_key)

            # Create a buffer for storing Left/Center/Right images from the same time
            if key not in buffer:
                buffer[key] = {
                    "Left": None,
                    "Center": None,
                    "Right": None
                }

            buffer[key][cam_pos] = {
                "created_time": created_time,
                "created_path": created_path,
                "monitor_dir": monitor_dir,
                "fname": Fname,
                "ext": Extension,
                "cam_pos": cam_pos,
            }
            print(f"[DISPATCHER][BUFFER][PUT] key={key}")
          
def processor(model, classes):
    """
    Runs inference once images from all three cameras have been collected in the buffer, 
    then passes the results to `process()`.
    """
    while True:
        # Process only keys with all three camera images
        for key, cams in list(buffer.items()):
            if all([cams["Left"], cams["Center"], cams["Right"]]):
                print(f"[PROCESSOR][READY] key={key}")
                items = [cams["Left"], cams["Center"], cams["Right"]]
                process(items, model, classes)
                del buffer[key]
        time.sleep(0.2)

def process(items, model, classes):
    """
    Performs YOLO inference on the three camera images,
    then saves the results to the database and creates the result images.
    """ 

    detect_products_cnt = []    
    img_paths = []

    # Save original image info
    for item in items:
        created_time = item["created_time"]
        img_path = item["created_path"]
        monitor_dir = item["monitor_dir"]
        Fname = item["fname"]
        Extension = item["ext"]
        cam_pos = item["cam_pos"]
        
        img_paths.append(img_path)
        
        insert_data = {
            "MONITOR_DIR": monitor_dir,
            "DATIME": created_time,
            "DANAME": Fname + Extension,
            "FILEPATH": inf_inp / f"{monitor_dir}" / f"{created_time.strftime('%Y-%m-%d')}",
            "REMARK": "",
        }
        try:
            insert_data["FILEPATH"].mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, insert_data["FILEPATH"] / insert_data["DANAME"])
            insert(ORG_IMG_TABLE, insert_data)
        except Exception as e:
            print(f"[PROCESSOR][ERROR] Save Original Image -> {e}")    
    
    # YOLO model inference    
    print(f"[INFERENCE][START]")
    with torch.no_grad():        
        results = model.predict(img_paths)

    print(f"[INFERENCE][END] total={len(results)}")

    detected_labels = []
    for pred in results:
        if pred is None or len(pred) == 0:
            continue
        for det in pred:
            x1, y1, x2, y2, conf, cls = det[:6]
            conf = float(conf)
            if conf < 0.9:
                continue
            cls = int(cls)
            label = model.names[cls]
            detected_labels.append(label)
            

    class_counts = dict(Counter(detected_labels))
    detect_products_cnt = [{"PRODUCTNAME": label, "PRODUCT_QTY": count} for label, count in class_counts.items()]

    # Handle exception: No objects detected
    if not class_counts: 
        detected_label = 'No Detection'            
        print(f"[INFERENCE] Results {Fname} -> No Detection")

        # Clean up temporary folders
        shutil.rmtree(pred_dir, ignore_errors=True)

    else:
        # Save monitoring summary to DB
        insert_monitering_data = {
            "DATIME": created_time,
            "MONITOR_DIR": monitor_dir,
            "DETECT_PRODUCTS": " | ".join(detected_labels),
            "ERROR_YN": "N",
            "REMARK": "",
        }
        
        try:
            insert(MONITOR_IMG_TABLE, insert_monitering_data)
        except Exception as e:
            print(f"[PROCESSOR][ERROR] Save Mointoring Data -> {e}")

        # Save detected products count to DB
        for product in detect_products_cnt:
            try:    
                # Get current quantity
                rows = select(PROD_CNT_TABLE, columns="PRODUCT_QTY", where="DATIME=%s AND PRODUCTNAME=%s",
                            params=(created_time.strftime('%Y-%m-%d'), product["PRODUCTNAME"]))
                if rows:
                    # Update quantity
                    current_qty = int(rows[0][0])
                    update(
                        PROD_CNT_TABLE,
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
                    insert(PROD_CNT_TABLE, insert_cnt_data)
            except Exception as e:
                print(f"[PROCESSOR][ERROR] Save Count of Products -> {e}")
        
    
if __name__ == '__main__':
    print(f"[SYSTEM][START] BASE_DIR -> {BASE_DIR}")
    print(f"[SYSTEM][START] SHARED -> {SHARED}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    INFER_INP_IMG_DIR.mkdir(parents=True, exist_ok=True)
    INFER_OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove previous predict folder
    shutil.rmtree(PREDICT_DIR, ignore_errors=True)
    print("[SYSTEM][INIT] Predict Directory Removed")

    # Load Model
    model_files = sorted(MODELS_DIR.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError("No .pth model found in models folder.")
    model_path = model_files[-1]

    print(f"[MODEL][LOAD] START: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model = MultiviewYOLODetector(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    print(f"[MODEL][LOAD] -> SUCCESS")
    
    # Load class names
    with open(CLASSES_TXT, 'r') as f:
        classes = f.read().splitlines()
    
    # THREAD START
    Thread(target=dispatcher, daemon=True).start()
    Thread(target=processor, args=(model, classes, ), daemon=True).start()
    print("[SYSTEM][THREAD] Dispatcher Started")
    print("[SYSTEM][THREAD] Processor Started")


    # WATCHDOG SETUP
    monitor_dir01 = Path("")
    monitor_dir02 = Path("")
    print("[SYSTEM][WATCHDOG] Monitoring Started")
    print(f"[SYSTEM][WATCHDOG] {monitor_dir01}")
    print(f"[SYSTEM][WATCHDOG] {monitor_dir02}")
    
    observer = Observer()
    observer.schedule(Handler("monitor_dir01", file_queue), str(monitor_dir01), recursive=True)
    observer.schedule(Handler("monitor_dir02", file_queue), str(monitor_dir02), recursive=True)

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
