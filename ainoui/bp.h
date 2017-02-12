#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define innode 91        //输入结点数，最后一个节点代表红黑，0是红，1是黑
#define hidenode 15      //隐含结点数
#define hidelayer 1     //隐含层数
#define outnode 1       //输出结点数
#define learningRate 0.9//学习速率，alpha

void addTrainData();
// --- -1~1 随机数产生器 --- 
inline double get_11Random()    // -1 ~ 1
{
	return ((2.0*(double)rand() / RAND_MAX) - 1);
}

// --- sigmoid 函数 --- 
inline double sigmoid(double x)
{
	double ans = 1 / (1 + exp(-x));
	return ans;
}

// --- 输入层节点。包含以下分量：--- 
// 1.value:     固定输入值； 
// 2.weight:    面对第一层隐含层每个节点都有权值； 
// 3.wDeltaSum: 面对第一层隐含层每个节点权值的delta值累积
typedef struct inputNode
{
	double value;
	vector<double> weight, wDeltaSum;
}inputNode;

// --- 输出层节点。包含以下数值：--- 
// 1.value:     节点当前值； 
// 2.delta:     与正确输出值之间的delta值； 
// 3.rightout:  正确输出值
// 4.bias:      偏移量
// 5.bDeltaSum: bias的delta值的累积，每个节点一个
typedef struct outputNode   // 输出层节点
{
	double value, delta, rightout, bias, bDeltaSum;
}outputNode;

// --- 隐含层节点。包含以下数值：--- 
// 1.value:     节点当前值； 
// 2.delta:     BP推导出的delta值；
// 3.bias:      偏移量
// 4.bDeltaSum: bias的delta值的累积，每个节点一个
// 5.weight:    面对下一层（隐含层/输出层）每个节点都有权值； 
// 6.wDeltaSum： weight的delta值的累积，面对下一层（隐含层/输出层）每个节点各自积累
typedef struct hiddenNode   // 隐含层节点
{
	double value, delta, bias, bDeltaSum;
	vector<double> weight, wDeltaSum;
}hiddenNode;

// --- 单个样本 --- 
typedef struct sample
{
	vector<double> in, out;
}sample;




// --- BP神经网络 --- 
class BpNet
{
public:
	BpNet();    //构造函数
	void forwardPropagationEpoc();  // 单个样本前向传播
	void backPropagationEpoc();     // 单个样本后向传播

	void training(static vector<sample> sampleGroup, double threshold, bool isContinue);// 更新 weight, bias
	void predict(vector<sample>& testGroup);                          // 神经网络预测

	void setInput(static vector<double> sampleIn);     // 设置学习样本输入
	void setOutput(static vector<double> sampleOut);    // 设置学习样本输出
	void importNet(char * fileName);//从文件导入网络
	void exportNet(char * fileName);//导出网络到文件

public:
	double error;
	inputNode* inputLayer[innode];                      // 输入层（仅一层）
	outputNode* outputLayer[outnode];                   // 输出层（仅一层）
	hiddenNode* hiddenLayer[hidelayer][hidenode];       // 隐含层（可能有多层）
};
class Train {
public:
	//子力价值，前红黑黑，分别为帅，士，相，马，车，炮，兵,黑的加负号
	bool isEval = false;
	const int power[7] = { 7,22, 20, 95, 210, 100,20 };
	int sampleCount;
	vector<sample> sampleGroup;//切记不要在全局变量中初始化vector!
	vector<sample> testGroup;
	BpNet evalNet;
	double inputData[90];//最后一个为红黑标记，0是红
public:
	void initTrainBp();
	void addTrainData();
	void trainingBp(bool isFinish);
	void predictEval();
	void importNet();
};