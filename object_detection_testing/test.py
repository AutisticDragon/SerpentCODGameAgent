import mss
Wd, Hd = 1920, 1080
ACTIVATION_RANGE = 300
YOLO_DIRECTORY = "models"
sct = mss.mss()

CONFIDENCE = 0.36
THRESHOLD = 0.22
ACTIVATION_RANGE = 400
labelsPath = os.path.sep.join([YOLO_DIRECTORY, "coco-dataset.labels"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
origbox = (int(Wd/2 - 250/2),
	int(Hd/2 - 500/2),
	int(Wd/2 + 250/2),
	int(Hd/2 + 500/2))


frame = np.array(sct.grab(region=origbox))
frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


if W is None or H is None:
            (H, W) = frame.shape[: 2]

frame = cv2.UMat(frame)
blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
	swapRB=False, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)
boxes = []
confidences = []
classIDs = []
for output in layerOutputs:
            for detection in output:
            	scores = detection[5:]
            	classID = 0
            	confidence = scores[classID]
            	if confidence > CONFIDENCE:
            		box = detection[0: 4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
if len(idxs) > 0:
	