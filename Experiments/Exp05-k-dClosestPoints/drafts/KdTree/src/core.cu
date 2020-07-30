#include "core.h"

struct Node {
    float* location;
    int axis;
    int index;
    Node* left;
    Node* right;
    Node() {}
};

struct sample {
    float* location;
    float axis_val;
    int axis;
    int index;
    sample() {}
};

int cmp(const void* p, const void* q)
{
    float l = ((sample*)p)->axis_val;
    float r = ((sample*)q)->axis_val;
    return l > r;
}

void copySample(sample* s1, sample* s2, int k)
{
    s2->location = (float*)malloc(k * sizeof(float));
    s2->axis = s1->index;
    s2->index = s1->index;
    s2->axis_val = s1->axis_val;

    for (int j = 0; j < k; j++) {
        s2->location[j] = s1->location[j];
    }
}

Node* buildKdTree(int k, int n, sample* referencePoints, int depth)
{
    if (!n) {
        return NULL;
    }

    int axis = depth % k;

    for (int i = 0; i < n; i++) {
        referencePoints[i].axis = axis;
        referencePoints[i].axis_val = referencePoints[i].location[axis];
    }

    qsort(referencePoints, n, sizeof(sample), cmp);


    Node* node = (Node*)malloc(sizeof(Node));
    node->location = (float*)malloc(k * sizeof(float));
    node->axis = axis;
    node->index = referencePoints[n / 2].index;
    for (int i = 0; i < k; i++) {
        node->location[i] = referencePoints[n / 2].location[i];
    }

    int median = n / 2;
    int left_size = median;
    int right_size = n - left_size - 1;
    sample* left_refer = (sample*)malloc(sizeof(sample) * left_size);
    sample* right_refer = (sample*)malloc(sizeof(sample) * right_size);


    int counter = 0;
    for (int i = 0; i < n; i++) {
        if (i == median) {
            counter = 0;
        }
        else if (i < median) {

            left_refer[counter].location = (float*)malloc(k * sizeof(float));
            left_refer[counter].axis = referencePoints[i].axis;
            left_refer[counter].index = referencePoints[i].index;
            left_refer[counter].axis_val = referencePoints[i].axis_val;

            for (int j = 0; j < k; j++) {
                left_refer[counter].location[j] = referencePoints[i].location[j];
            }

            counter++;
        }
        else if (i > median) {
            right_refer[counter].location = (float*)malloc(k * sizeof(float));
            right_refer[counter].axis = referencePoints[i].axis;
            right_refer[counter].index = referencePoints[i].index;
            right_refer[counter].axis_val = referencePoints[i].axis_val;

            for (int j = 0; j < k; j++) {
                right_refer[counter].location[j] = referencePoints[i].location[j];
            }

            counter++;
        }
    }

    for (int i = 0; i < n; i++) {
        free(referencePoints[i].location);
    }
    free(referencePoints);

    node->left = buildKdTree(k, left_size, left_refer, depth + 1);
    node->right = buildKdTree(k, right_size, right_refer, depth + 1);

    return node;
}

void printKdTree(Node* root, int k)
{
    int counter = 1, next = 2;
    std::queue<Node*> q;
    q.push(root);
    while (!q.empty()) {
        Node* tmp = q.front();
        q.pop();
        counter--;
        if (!tmp) {
            printf("NULL NULL NULL ");
            printf(counter ? " | " : "\n");
            if (!counter) {
                counter = next;
                next = 2 * next;
            }
            continue;
        }
        for (int j = 0; j < k; j++) {
            printf("%f ", tmp->location[j]);
        }
        printf("(%d, axis=%d)", tmp->index, tmp->axis);
        printf(counter ? " | " : "\n");
        if (!counter) {
            counter = next;
            next = 2 * next;
        }
        q.push(tmp->left);
        q.push(tmp->right);
    }
    puts("");
}

void kdTreeFindNearest(Node* root, int k, int n, int search_bias, float* visited,
                       float* searchPoints, float* min_dist, Node** nearest, int* nearest_index)
{
    int pos = -1, axis;
    float minDist, dist, diff;
    Node** search_path = (Node**)malloc(sizeof(Node*) * n);
    float* sub_search = searchPoints + search_bias;

    Node* ptr = root;
    while (ptr) {
        search_path[++pos] = ptr;
        if (sub_search[ptr->axis] <= ptr->location[ptr->axis]) {
            ptr = ptr->left;
        }
        else {
            ptr = ptr->right;
        }
    }

    *nearest = search_path[pos];

    if (!visited[root->index]) {
        dist = 0;
        for (int i = 0; i < k; i++) {
            diff = sub_search[i] - (*nearest)->location[i];
            dist += (diff * diff);
        }
        visited[root->index] = dist;
        minDist = dist;
    }
    else {
        minDist = visited[root->index];
    }


    while (pos >= 0) {
        ptr = search_path[pos--];
        axis = ptr->axis;

        if (!ptr->left && !ptr->right) {

            if (!visited[root->index]) {
                dist = 0;
                for (int i = 0; i < k; i++) {
                    diff = sub_search[i] - (*nearest)->location[i];
                    dist += (diff * diff);
                }
                visited[root->index] = dist;
            }
            else {
                dist = visited[root->index];
            }
            if (minDist > dist) {
                minDist = dist;
                *nearest = ptr;
            }

        }
        else if (fabs(ptr->location[axis] - sub_search[axis]) < sqrt(minDist)) {

            if (!visited[root->index]) {
                dist = 0;
                for (int i = 0; i < k; i++) {
                    diff = sub_search[i] - (*nearest)->location[i];
                    dist += (diff * diff);
                }
                visited[root->index] = dist;
            }
            else {
                dist = visited[root->index];
            }
            if (minDist > dist) {
                minDist = dist;
                *nearest = ptr;
            }

            if (sub_search[axis] <= ptr->location[axis]) {
                if (ptr->right) {
                    float new_dist = 0;
                    Node** new_nearest = (Node**)malloc(sizeof(Node*));
                    kdTreeFindNearest(ptr->right, k, n, search_bias, visited, searchPoints, &new_dist, new_nearest, nearest_index);
                    if (minDist > new_dist) {
                        minDist = new_dist;
                        *nearest = *new_nearest;
                    }
                    free(new_nearest);
                }
            }
            else {
                if (ptr->left) {
                    float new_dist = 0;
                    Node** new_nearest = (Node**)malloc(sizeof(Node*));
                    kdTreeFindNearest(ptr->left, k, n, search_bias, visited, searchPoints, &new_dist, new_nearest, nearest_index);
                    if (minDist > new_dist) {
                        minDist = new_dist;
                        *nearest = *new_nearest;
                    }
                    free(new_nearest);
                }
            }

        }
    }
    *min_dist = minDist;
    *nearest_index = (*nearest)->index;
    free(search_path);
}

void FindNearest(Node* root, int k, int n, int search_bias,
                 float* searchPoints, int* indices)
{
    float min_dist = 0;
    int tmp;
    Node** ptr = (Node**)malloc(sizeof(Node*));
    float* visited = (float*)malloc(sizeof(float) * n);
    kdTreeFindNearest(root, k, n, search_bias, visited, searchPoints, &min_dist, ptr, &tmp);
    indices[search_bias / k] = tmp;
}

extern void cudaCallback(int k, int m, int n, float* searchPoints,
                         float* referencePoints, int** results)
{
    sample* reference = (sample*)malloc(sizeof(sample) * size_t(n));

    for (int i = 0; i < n; i++) {
        reference[i].location = (float*)malloc(k * sizeof(float));
        reference[i].axis = 0;
        reference[i].index = i;

        for (int j = 0; j < k; j++) {
            reference[i].location[j] = referencePoints[i * k + j];
        }

        reference[i].axis_val = reference[i].location[0];

    }

    Node* root = buildKdTree(k, n, reference, 0);

    // printKdTree(root, k);

    int* tmp = (int*)malloc(sizeof(int) * m);

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        FindNearest(root, k, n, mInd * k, searchPoints, tmp);
    }

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}