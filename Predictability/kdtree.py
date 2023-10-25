import numpy as np
import gc

class KDNode(object):
    def __init__(
            self, features,
            left=None, right=None,
            axis_no=0, depth=0
    ):
        # 每个节点的特征值
        self.features = features
        # 节点的左孩子
        self.left = left
        # 节点的右孩子
        self.right = right
        # 划分维度编号
        self.axis_no = axis_no
        # 节点所在的深度
        self.depth = depth


class KDTree(object):
    def __init__(self, n_features):
        # 根节点
        self.root = None
        # 记录维度，总共有多少个特征
        self.dimensions = n_features

    def getAxis(self, depth: int, n_features: int, base: int) -> int:
        return (depth+base) % n_features

    '''
    Function:
    ----------
    按照指定列对矩阵进行排序

    Parameters
    ----------
    depth : int
            当前构造节点的深度
    n_features : int
            数据集中特征的数量

    Returns
    -------
    axis_number : int
            第depth层应该在第.axis_number 划分数据

    Notes
    -------      
        对于有n_features个特征的数据集，
        深度为j的节点，选择X(L)为切分坐标轴，
        L=j mod k

    Examples
    -------
    a = np.array([
        [300, 1, 10, 999],
        [200, 2, 30, 987],
        [100, 3, 10, 173]
    ])
    sort_index = np.argsort(a, axis=0)
    print(sort_index)
    >>>
        [[2 0 0 2]
         [1 1 2 1]
         [0 2 1 0]]

    sort_index[:, 3]
    >>> 
        [2 1 0]
    print(a[sort_index[:, 3]])
    >>>
        [[100   3  10 173]
         [200   2  30 987]
         [300   1  10 999]]
    '''

    def getSortDataset(self, dataset, axis_no):
        # 将矩阵按列排序，获取每一列升序结果的索引，结果仍为一个矩阵
        sort_index = np.argsort(dataset, axis=0)
        return dataset[sort_index[:, axis_no]]

    '''
        Function:
        ----------
        构造KD树

        Parameters
        ----------
        depth : int
                当前构造节点的深度
        dataset : numpy.ndarray
                只包含特征的矩阵

        Returns
        -------
                构造好的KDTree

        Notes
        -------
        1. 如果数据集中只有一条数据，则赋予空的叶子节点
        2. 如果不止一条数据，则进行如下操作：
            a. 根据构造树当前的深度，选定划分轴（根据哪个特征进行划分）
            b. 根据划分轴（该特征），对数据集按照该特征从小到大排序
            c. 选出中位数、排序特征中大于、小于中位数的子数据集
            d. 递归调用自身，构造KDTree
        '''

    def create(self, feature_dataset, depth, base=0):
        samples = feature_dataset.shape[0]
        if samples < 1:
            return None
        if samples == 1:
            new_node = KDNode(
                feature_dataset[0],
                depth=depth
            )
        else:
            # 获取分隔坐标轴编号
            axis_no = self.getAxis(depth, self.dimensions, base)
            # 获取按第 axis_no 轴排好序的矩阵
            sorted_dataset = self.getSortDataset(feature_dataset, axis_no)
            # 获取第 axis_no 轴的中位数
            median_no = samples // 2
            # 获取需要设置在左子树的数据集及标签
            left_dataset = sorted_dataset[: median_no, :]
            # 获取需要设置在右子树的数据集及标签
            right_dataset = sorted_dataset[median_no + 1:, :]
            # 获取分割点
            mid_point = sorted_dataset[median_no, :]
            # 构造KDTree的节点
            del feature_dataset
            del sorted_dataset
            new_node = KDNode(
                mid_point,
                axis_no=axis_no,
                depth=depth
            )
            # 构造左子树与右子树
            new_node.left = self.create(
                left_dataset,
                depth + 1,
                base
            )
            new_node.right = self.create(
                right_dataset,
                depth + 1,
                base
            )
        return new_node

    def getDepth(self, node: KDNode):
        if node is None:
            return 0
        else:
            return max(
                self.getDepth(node.left),
                self.getDepth(node.right)
            ) + 1

    def deep_search(self, node, ranges):
        if node is None:
            return 0
        num = 1
        for i in range(self.dimensions):
            if ranges[i][0] <= node.features[i] <= ranges[i][1]:
                continue
            else:
                num = 0
                break

        left = 0
        right = 0
        if node.features[node.axis_no] >= ranges[node.axis_no][0]:
            left = self.deep_search(node.left, ranges)
        if node.features[node.axis_no] <= ranges[node.axis_no][1]:
            right = self.deep_search(node.right, ranges)
        return num + left + right

    def range_search(self, ranges):
        return self.deep_search(self.root, ranges)

    def visualize(self, node):
        if node is None:
            return
        print("depth:%d, axis:%d, features:%s" % (node.depth, node.axis_no, str(node.features)))
        self.visualize(node.left)
        self.visualize(node.right)


