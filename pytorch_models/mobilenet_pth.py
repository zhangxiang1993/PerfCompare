from mobilenet import *
model = MobileNet()
PATH = "D:\\PerfCompare\\pytorch_models\\mobilenet_v1_1.0_224.pth"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['state_dict'])


model.eval()