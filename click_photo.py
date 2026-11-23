import cv2
import json

image_path = "potatoes_2.jpg"
img = cv2.imread(image_path)
points = []

def redraw():
    canvas = img.copy()
    for i, (x, y) in enumerate(points, 1):
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
    cv2.putText(canvas, f"Count: {len(points)} | LMB: add, RMB: undo, S: save, Q: quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Potato Annotator", canvas)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()
        redraw()

cv2.namedWindow("Potato Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Potato Annotator", mouse_callback)
redraw()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        with open("potato_points.json", "w") as f:
            json.dump({"image": image_path, "count": len(points), "points": points}, f, indent=2)
        print(f"💾 Сохранено {len(points)} точек в potato_points.json")
    elif key == ord("q") or key == 27:  # Q или Esc для выхода
        break

cv2.destroyAllWindows()
