from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO

class MultiviewYOLODetector(nn.Module):
    def __init__(self, pretrained_model='yolov8n.pt', num_classes=5, lr=1e-4):
        super(MultiviewYOLODetector, self).__init__()
        
        # 1. 공통 백본 (특징 추출기)
        # YOLOv8의 주 구조에서 특징을 결합하기 위해 백본만 추출합니다.
        base_yolo = YOLO(pretrained_model).model
        self.backbone = nn.Sequential(*list(base_yolo.model[:10])) # 이미지의 고차원 특징 추출
        self.detect = base_yolo.model[-1]      # Detect
        self.neck = nn.Sequential(*list(base_yolo.model[10:-1]))  # Neck


        # 2. 특징 융합 레이어 (Feature Fusion)
        # 3개 시점의 특징이 합쳐지므로 채널 수가 3배가 됩니다. (YOLOv8n의 백본 끝 채널이 256일 경우 -> 768)
        # 여기서는 임시 포워딩을 통해 채널 크기를 동적으로 알아냅니다.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            dummy_feat = self.backbone(dummy)
            in_channels = dummy_feat.size(1) # 한 시점의 채널 수
            
        # 3개 시점의 채널을 결합한 후, 다시 원래 YOLO 헤드가 처리할 수 있는 채널 수로 압축하는 합성곱 층
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        
        # 3. 예측을 담당할 YOLO Neck + Head 
        # 융합된 특징 맵을 받아 최종적으로 바운딩 박스와 클래스를 뽑아냅니다.
        self.detection_head = nn.Sequential(*list(base_yolo.model[10:]))
        
        # 4. 내장 옵티마이저
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, imgs):
        """imgs : (img_left, img_center, img_right)"""

        img_left, img_center, img_right = imgs

        p3_l, p4_l, p5_l = self.backbone(img_left)
        p3_c, p4_c, p5_c = self.backbone(img_center)
        p3_r, p4_r, p5_r = self.backbone(img_right)

        fused_p3 = self.fuse_p3(torch.cat((p3_l, p3_c, p3_r), dim=1))
        fused_p4 = self.fuse_p4(torch.cat((p4_l, p4_c, p4_r), dim=1))
        fused_p5 = self.fuse_p5(torch.cat((p5_l, p5_c, p5_r), dim=1))

        neck_feats = self.neck([fused_p3, fused_p4, fused_p5])

        preds = self.detect(neck_feats)

        return preds
    
    def train_step(self, imgs, labels):
        self.train()
        self.optimizer.zero_grad()

        preds = self.forward(imgs)
        loss, loss_items = self.detect.criterion(preds, labels)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_items": loss_items
        }

    def evaluate_step(self, imgs, labels):
        self.eval()

        with torch.no_grad():

            preds = self.forward(imgs)
            loss, loss_items = self.detect.criterion(preds, labels)
            decoded = self.detect.predict_by_feat(preds)

        return {
            "val_loss": loss.item(),
            "loss_items": loss_items,
            "preds": decoded
        }

    def predict(self, imgs):
        self.eval()
        with torch.no_grad():
            preds = self.forward(imgs)
            decoded = self.detect.predict_by_feat(preds)
        return decoded