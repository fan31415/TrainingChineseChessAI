#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define innode 91        //�������������һ���ڵ�����ڣ�0�Ǻ죬1�Ǻ�
#define hidenode 15      //���������
#define hidelayer 1     //��������
#define outnode 1       //��������
#define learningRate 0.9//ѧϰ���ʣ�alpha

void addTrainData();
// --- -1~1 ����������� --- 
inline double get_11Random()    // -1 ~ 1
{
	return ((2.0*(double)rand() / RAND_MAX) - 1);
}

// --- sigmoid ���� --- 
inline double sigmoid(double x)
{
	double ans = 1 / (1 + exp(-x));
	return ans;
}

// --- �����ڵ㡣�������·�����--- 
// 1.value:     �̶�����ֵ�� 
// 2.weight:    ��Ե�һ��������ÿ���ڵ㶼��Ȩֵ�� 
// 3.wDeltaSum: ��Ե�һ��������ÿ���ڵ�Ȩֵ��deltaֵ�ۻ�
typedef struct inputNode
{
	double value;
	vector<double> weight, wDeltaSum;
}inputNode;

// --- �����ڵ㡣����������ֵ��--- 
// 1.value:     �ڵ㵱ǰֵ�� 
// 2.delta:     ����ȷ���ֵ֮���deltaֵ�� 
// 3.rightout:  ��ȷ���ֵ
// 4.bias:      ƫ����
// 5.bDeltaSum: bias��deltaֵ���ۻ���ÿ���ڵ�һ��
typedef struct outputNode   // �����ڵ�
{
	double value, delta, rightout, bias, bDeltaSum;
}outputNode;

// --- ������ڵ㡣����������ֵ��--- 
// 1.value:     �ڵ㵱ǰֵ�� 
// 2.delta:     BP�Ƶ�����deltaֵ��
// 3.bias:      ƫ����
// 4.bDeltaSum: bias��deltaֵ���ۻ���ÿ���ڵ�һ��
// 5.weight:    �����һ�㣨������/����㣩ÿ���ڵ㶼��Ȩֵ�� 
// 6.wDeltaSum�� weight��deltaֵ���ۻ��������һ�㣨������/����㣩ÿ���ڵ���Ի���
typedef struct hiddenNode   // ������ڵ�
{
	double value, delta, bias, bDeltaSum;
	vector<double> weight, wDeltaSum;
}hiddenNode;

// --- �������� --- 
typedef struct sample
{
	vector<double> in, out;
}sample;




// --- BP������ --- 
class BpNet
{
public:
	BpNet();    //���캯��
	void forwardPropagationEpoc();  // ��������ǰ�򴫲�
	void backPropagationEpoc();     // �����������򴫲�

	void training(static vector<sample> sampleGroup, double threshold, bool isContinue);// ���� weight, bias
	void predict(vector<sample>& testGroup);                          // ������Ԥ��

	void setInput(static vector<double> sampleIn);     // ����ѧϰ��������
	void setOutput(static vector<double> sampleOut);    // ����ѧϰ�������
	void importNet(char * fileName);//���ļ���������
	void exportNet(char * fileName);//�������絽�ļ�

public:
	double error;
	inputNode* inputLayer[innode];                      // ����㣨��һ�㣩
	outputNode* outputLayer[outnode];                   // ����㣨��һ�㣩
	hiddenNode* hiddenLayer[hidelayer][hidenode];       // �����㣨�����ж�㣩
};
class Train {
public:
	//������ֵ��ǰ��ںڣ��ֱ�Ϊ˧��ʿ���࣬�������ڣ���,�ڵļӸ���
	bool isEval = false;
	const int power[7] = { 7,22, 20, 95, 210, 100,20 };
	int sampleCount;
	vector<sample> sampleGroup;//�мǲ�Ҫ��ȫ�ֱ����г�ʼ��vector!
	vector<sample> testGroup;
	BpNet evalNet;
	double inputData[90];//���һ��Ϊ��ڱ�ǣ�0�Ǻ�
public:
	void initTrainBp();
	void addTrainData();
	void trainingBp(bool isFinish);
	void predictEval();
	void importNet();
};