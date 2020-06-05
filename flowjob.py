import complexity as cmplx
import connectivity_by_contour as conec_cube
import connectivity_by_contourColor as conec_antr
import contourShapeDetect as cts
import symmetrical_by_hist as syt
import whRatio as wh
import element_array as ea



class flow_model:

    def inference_inet(self, img, gy):
        result = []
        img = [img]
        # 复杂度
        res_complex = cmplx.compJudge(img, 0, gy)
        result.append(res_complex)
        #
        # 轮廓形状
        cont_shape = cts.clasConShape(img, 0, gy)
        result.append(cont_shape)

        # 对称
        symmetric = syt.judgeSym(img, 0, gy)
        result.append(symmetric)

        # 扁瘦
        wOrh = wh.classifyRatio(img, 0, gy)
        result.append(wOrh)

        # 阵列
        ifArr = ea.classifyArray(img, 0, gy)
        result.append(ifArr)
        return result


    def inference_pz1(self, img,round_ori,rouCube_ori):
        result = []
        img = [img]
        # 碰撞镂空
        cnect = conec_cube.judgeCon(img, 0,round_ori,rouCube_ori )
        result.append(cnect)
        return result

    def inference_pz2(self, img, formal_triangle_npy,left_triangle_npy ):
        result = []
        img = [img]
        # 三角镂空
        cnect_antr = conec_antr.judgeCon(img, 0, formal_triangle_npy,left_triangle_npy )
        result.append(cnect_antr)
        return result




























