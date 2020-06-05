import cv2
import numpy as np

path = "../../data/testsym/ori/10068.png"

# for i in range(10):
#     for j in ['a','b','c']:
#         if i == 2 and j == 'b':
#             break
#         print(str(i)+str(j))
#     else:
#         continue
#     break


src = cv2.imread(path)
print(src.shape)
Z = src.reshape((-1, 3))
print(Z.shape)
Z = np.float32(Z)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# criteria=(cv2.TERM_CRITERIA_EPS , 100, 1)
# ret, label, center = cv2.kmeans(Z, 2, None,
#                                 criteria,
#                                 10,  cv2.KMEANS_RANDOM_CENTERS)
ret, label, center = cv2.kmeans(Z,  5, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

print(ret)
print(label)
print(center)

label = label.flatten()
print(label)

center = np.uint8(center)
print(center)

result = center[label]
result = result.reshape(src.shape)
print(result)
print(result.shape)


cv2.imshow('a',src)
cv2.waitKey(0)
cv2.imshow('b',result)
cv2.waitKey(0)












