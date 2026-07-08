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

    def forward(self, img_left, img_center, img_right):
        """세 개의 이미지를 동시에 보고 하나의 종합 예측 공간으로 융합합니다."""
        # 각 시점별 고차원 특징 맵 추출
        feat_l = self.backbone(img_left)
        feat_c = self.backbone(img_center)
        feat_r = self.backbone(img_right)
        
        # [핵심] 채널 차원(dim=1)을 기준으로 세 시점의 특징을 하나로 결합 (Concat)
        combined_feat = torch.cat((feat_l, feat_c, feat_r), dim=1)
        
        # 융합 레이어를 통과시켜 연관 관계 학습 및 채널 압축
        fused_feat = self.fusion_layer(combined_feat)
        
        # 융합된 하나의 특징 맵을 YOLO의 디텍션 헤드에 입력하여 최종 결과 도출
        # (이 단계에서 좌/중앙/우의 시각적 정보가 섞인 상태로 박스를 찾게 됩니다)
        output = self.detection_head(fused_feat)
        return output

    def train_step(self, integerated_img, integrated_label):
        """
        융합된 단일 예측값과 통합 라벨을 비교하여 학습합니다.
        integrated_label: 세 시점의 객체 정보가 하나의 이미지 좌표계(또는 가상 공간)로 통합된 정답 데이터
        """
        self.train()
        self.optimizer.zero_grad()
        
        # 세 시점을 동시에 보고 뱉은 하나의 종합 예측값
        img_left, img_center, img_right = integerated_img
        fused_output = self.forward(img_left, img_center, img_right)
        
        # 통합된 단일 Loss 계산 (YOLOv8 내부 크라이테리언 활용)
        # 이제 모델은 3개의 시점을 따로 보는 게 아니라, '종합 정보'와 '통합 정답'을 비교합니다.
        loss, loss_items = self.detection_head[-1].criterion(fused_output, integrated_label)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate_step(self, integrated_img, integrated_label):
        """
        [수정된 검증 단계]
        3개 시점을 동시에 입력받아 융합된 단일 결과와 
        통합 레이블(integrated_label)을 비교하여 검증 Loss를 산출합니다.
        """
        self.eval()
        with torch.no_grad():
            img_left, img_center, img_right = integrated_img
            fused_output = self.forward(img_left, img_center, img_right)
            
            # 1. Loss 계산
            loss, _ = self.detection_head[-1].criterion(fused_output, integrated_label)
            
            # 2. 예측값 추출 (메트릭 계산을 위해)
            preds = self.detection_head[-1].predict_by_feat(fused_output)
            
            return {
                "val_loss": loss.item(),
                "preds": preds # 예측 결과도 함께 반환
            }

    def predict(self, integerated_img):
        """
        [추론 단계]
        세 각도의 이미지를 동시에 찔러 넣으면, 
        이를 하나로 종합 연산하여 최종 통합 바운딩 박스와 클래스 결과를 뽑아냅니다.
        """
        self.eval()
        with torch.no_grad():
            # forward에서 이미 융합 연산이 끝나 한 세트의 출력만 나옵니다.
            img_left, img_center, img_right = integerated_img
            fused_output = self.forward(img_left, img_center, img_right)
            
            # YOLOv8 디코더 헤드를 거친 최종 Raw 예측 텐서 추출
            # (일반적으로 NMS 전단계의 예측 원본 데이터 [Batch, 횡/종/높이 등의 채널, 앵커수] 구조)
            final_raw_preds = fused_output[0] if isinstance(fused_output, tuple) else fused_output
            
        return {
            # 이제 각 시점별로 쪼개진 결과가 아니라, 세 각도가 종합 반영된 단 하나의 통합 예측 배열이 나옵니다.
            "integrated_predictions": final_raw_preds.cpu().numpy()
        }