#include "bp.h"
#include <Windows.h>
using namespace std;

BpNet::BpNet()
{
	srand((unsigned)time(NULL));        // 随机数种子    
	error = 100.f;                      // error初始值，极大值即可

										// 初始化输入层
	for (int i = 0; i < innode; i++)
	{
		inputLayer[i] = new inputNode();
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->weight.push_back(get_11Random());
			inputLayer[i]->wDeltaSum.push_back(0.f);
		}
	}

	// 初始化隐藏层
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->weight.push_back(get_11Random());
					hiddenLayer[i][j]->wDeltaSum.push_back(0.f);
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j] = new hiddenNode();
				hiddenLayer[i][j]->bias = get_11Random();
				for (int k = 0; k < hidenode; k++) { hiddenLayer[i][j]->weight.push_back(get_11Random()); }
			}
		}
	}

	// 初始化输出层
	for (int i = 0; i < outnode; i++)
	{
		outputLayer[i] = new outputNode();
		outputLayer[i]->bias = get_11Random();
	}
}

void BpNet::forwardPropagationEpoc()
{
	// forward propagation on hidden layer
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < innode; k++)
				{
					sum += inputLayer[k]->value * inputLayer[k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k < hidenode; k++)
				{
					sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
				}
				sum += hiddenLayer[i][j]->bias;
				hiddenLayer[i][j]->value = sigmoid(sum);
			}
		}
	}

	// forward propagation on output layer
	for (int i = 0; i < outnode; i++)
	{
		double sum = 0.f;
		for (int j = 0; j < hidenode; j++)
		{
			sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
		}
		sum += outputLayer[i]->bias;
		outputLayer[i]->value = sigmoid(sum);
	}
}

void BpNet::backPropagationEpoc()
{
	// backward propagation on output layer
	// -- compute delta
	for (int i = 0; i < outnode; i++)
	{
		double tmpe = fabs(outputLayer[i]->value - outputLayer[i]->rightout);
		error += tmpe * tmpe / 2;

		outputLayer[i]->delta
			= (outputLayer[i]->value - outputLayer[i]->rightout)*(1 - outputLayer[i]->value)*outputLayer[i]->value;
	}

	// backward propagation on hidden layer
	// -- compute delta
	for (int i = hidelayer - 1; i >= 0; i--)    // 反向计算
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k<outnode; k++) { sum += outputLayer[k]->delta * hiddenLayer[i][j]->weight[k]; }
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				double sum = 0.f;
				for (int k = 0; k<hidenode; k++) { sum += hiddenLayer[i + 1][k]->delta * hiddenLayer[i][j]->weight[k]; }
				hiddenLayer[i][j]->delta = sum * (1 - hiddenLayer[i][j]->value) * hiddenLayer[i][j]->value;
			}
		}
	}

	// backward propagation on input layer
	// -- update weight delta sum
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode; j++)
		{
			inputLayer[i]->wDeltaSum[j] += inputLayer[i]->value * hiddenLayer[0][j]->delta;
		}
	}

	// backward propagation on hidden layer
	// -- update weight delta sum & bias delta sum
	for (int i = 0; i < hidelayer; i++)
	{
		if (i == hidelayer - 1)
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < outnode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * outputLayer[k]->delta;
				}
			}
		}
		else
		{
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->bDeltaSum += hiddenLayer[i][j]->delta;
				for (int k = 0; k < hidenode; k++)
				{
					hiddenLayer[i][j]->wDeltaSum[k] += hiddenLayer[i][j]->value * hiddenLayer[i + 1][k]->delta;
				}
			}
		}
	}

	// backward propagation on output layer
	// -- update bias delta sum
	for (int i = 0; i < outnode; i++) outputLayer[i]->bDeltaSum += outputLayer[i]->delta;
}
const double deltaMin = 0.0002;
void BpNet::training(static vector<sample> sampleGroup, double threshold, bool isContinue)
{
	double lastError = 1;
	DWORD lastTime = GetTickCount();

	if (isContinue) {
		importNet("tempNet.txt");
	}
	/**
	if (fopen("tempNet.txt", "r") != NULL) {
	importNet("tempNet.txt");
	}*/
	int sampleNum = sampleGroup.size();

	while (error > threshold)
		//for (int curTrainingTime = 0; curTrainingTime < trainingTime; curTrainingTime++)
	{
		//cout << "training error: " << error << endl;
		error = 0.f;

		// initialize delta sum
		for (int i = 0; i < innode; i++) inputLayer[i]->wDeltaSum.assign(inputLayer[i]->wDeltaSum.size(), 0.f);
		for (int i = 0; i < hidelayer; i++) {
			for (int j = 0; j < hidenode; j++)
			{
				hiddenLayer[i][j]->wDeltaSum.assign(hiddenLayer[i][j]->wDeltaSum.size(), 0.f);
				hiddenLayer[i][j]->bDeltaSum = 0.f;
			}
		}
		for (int i = 0; i < outnode; i++) outputLayer[i]->bDeltaSum = 0.f;

		for (int iter = 0; iter < sampleNum; iter++)
		{
			setInput(sampleGroup[iter].in);
			setOutput(sampleGroup[iter].out);

			forwardPropagationEpoc();
			backPropagationEpoc();
		}

		// backward propagation on input layer
		// -- update weight
		for (int i = 0; i < innode; i++)
		{
			for (int j = 0; j < hidenode; j++)
			{
				inputLayer[i]->weight[j] -= learningRate * inputLayer[i]->wDeltaSum[j] / sampleNum;
			}
		}

		// backward propagation on hidden layer
		// -- update weight & bias
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == hidelayer - 1)
			{
				for (int j = 0; j < hidenode; j++)
				{
					// bias
					hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
					for (int k = 0; k < outnode; k++)
					{
						hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					// bias
					hiddenLayer[i][j]->bias -= learningRate * hiddenLayer[i][j]->bDeltaSum / sampleNum;

					// weight
					for (int k = 0; k < hidenode; k++)
					{
						hiddenLayer[i][j]->weight[k] -= learningRate * hiddenLayer[i][j]->wDeltaSum[k] / sampleNum;
					}
				}
			}
		}

		// backward propagation on output layer
		// -- update bias
		for (int i = 0; i < outnode; i++)
		{
			outputLayer[i]->bias -= learningRate * outputLayer[i]->bDeltaSum / sampleNum;
		}
		
		DWORD now = GetTickCount();
		if (now - lastTime >= 30000) {
			lastTime = now;
			//cout << "save temp network\n";
			cout << "training error: " << error << endl;
			
			

			exportNet("tempNet.txt");

			if (lastError - error < deltaMin) {
				//cout << "so slow!\n";
				return;
			}
			else {
				lastError = error;
			}
		}
	}
	exportNet("net.txt");
	/**
	if (remove("tempNet.txt") == -1) {
	printf("delete error\n");
	perror("remove");
	}*/
}
void BpNet::importNet(char * fileName) {
	FILE *pFile = fopen(fileName, //打开文件的名称
		"r"); // 文件打开方式 如果原来有内容也会销毁
	vector<double>::iterator iter;
	for (int i = 0; i < innode; i++) {
		for (int j = 0; j < hidenode; j++) {
			fscanf(pFile, "%lf", &inputLayer[i]->weight[j]);
		}
	}
	for (int i = 0; i < outnode; i++) {
		fscanf(pFile, "%lf", &outputLayer[i]->bias);;
	}
	for (int i = 0; i < hidenode; i++) {
		fscanf(pFile, "%lf", &hiddenLayer[0][i]->bias);
		for (int k = 0; k < outnode; k++)
		{
			fscanf(pFile, "%lf", &hiddenLayer[0][i]->weight[k]);
		}


	}
	fclose(pFile);
}
void BpNet::exportNet(char * fileName) {
	//持久化
	FILE *pFile = fopen(fileName, //打开文件的名称
		"w"); // 文件打开方式 如果原来有内容也会销毁

			  //同时序列化
	vector<double>::iterator iter;
	for (int i = 0; i < innode; i++) {
		for (iter = inputLayer[i]->weight.begin(); iter != inputLayer[i]->weight.end(); iter++)
		{
			fprintf(pFile, "%lf ", *iter);
		}
	}
	//fprintf(pFile, "input\n");
	for (int i = 0; i < outnode; i++) {
		fprintf(pFile, "%lf ", outputLayer[i]->bias);;
	}
	//fprintf(pFile, "output\n");
	for (int i = 0; i < hidenode; i++) {
		fprintf(pFile, "%lf ", hiddenLayer[0][i]->bias);
		//fprintf(pFile, "bias ");
		for (iter = hiddenLayer[0][i]->weight.begin(); iter != hiddenLayer[0][i]->weight.end(); iter++)
		{
			fprintf(pFile, "%lf ", *iter);
		}
		//fprintf(pFile, "weight ");
	}
	fclose(pFile);
}

void BpNet::predict(vector<sample>& testGroup)
{
	int testNum = testGroup.size();

	for (int iter = 0; iter < testNum; iter++)
	{
		testGroup[iter].out.clear();
		setInput(testGroup[iter].in);

		// forward propagation on hidden layer
		for (int i = 0; i < hidelayer; i++)
		{
			if (i == 0)
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < innode; k++)
					{
						sum += inputLayer[k]->value * inputLayer[k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = sigmoid(sum);
				}
			}
			else
			{
				for (int j = 0; j < hidenode; j++)
				{
					double sum = 0.f;
					for (int k = 0; k < hidenode; k++)
					{
						sum += hiddenLayer[i - 1][k]->value * hiddenLayer[i - 1][k]->weight[j];
					}
					sum += hiddenLayer[i][j]->bias;
					hiddenLayer[i][j]->value = sigmoid(sum);
				}
			}
		}

		// forward propagation on output layer
		for (int i = 0; i < outnode; i++)
		{
			double sum = 0.f;
			for (int j = 0; j < hidenode; j++)
			{
				sum += hiddenLayer[hidelayer - 1][j]->value * hiddenLayer[hidelayer - 1][j]->weight[i];
			}
			sum += outputLayer[i]->bias;
			outputLayer[i]->value = sigmoid(sum);
			testGroup[iter].out.push_back(outputLayer[i]->value);
		}
	}
}

void BpNet::setInput(static vector<double> sampleIn)
{
	for (int i = 0; i < innode; i++) inputLayer[i]->value = sampleIn[i];
}

void BpNet::setOutput(static vector<double> sampleOut)
{
	for (int i = 0; i < outnode; i++) outputLayer[i]->rightout = sampleOut[i];
}

int mains()
{
	BpNet testNet;

	// 学习样本
	vector<double> samplein[4];
	vector<double> sampleout[4];
	samplein[0].push_back(0); samplein[0].push_back(0); sampleout[0].push_back(0);
	samplein[1].push_back(0); samplein[1].push_back(1); sampleout[1].push_back(1);
	samplein[2].push_back(1); samplein[2].push_back(0); sampleout[2].push_back(1);
	samplein[3].push_back(1); samplein[3].push_back(1); sampleout[3].push_back(0);
	sample sampleInOut[4];
	for (int i = 0; i < 4; i++)
	{
		sampleInOut[i].in = samplein[i];
		sampleInOut[i].out = sampleout[i];
	}
	vector<sample> sampleGroup(sampleInOut, sampleInOut + 4);
	testNet.training(sampleGroup, 0.00001,true);

	//testNet.importNet("net.txt");
	// 测试数据
	vector<double> testin[4];
	vector<double> testout[4];
	testin[0].push_back(0.1);   testin[0].push_back(0.2);
	testin[1].push_back(1);  testin[1].push_back(1);
	testin[2].push_back(1);   testin[2].push_back(0);
	testin[3].push_back(0);  testin[3].push_back(1);
	sample testInOut[4];
	for (int i = 0; i < 4; i++) testInOut[i].in = testin[i];
	vector<sample> testGroup(testInOut, testInOut + 4);

	// 预测测试数据，并输出结果
	testNet.predict(testGroup);
	for (int i = 0; i < testGroup.size(); i++)
	{
		for (int j = 0; j < testGroup[i].in.size(); j++) cout << testGroup[i].in[j] << "\t";
		cout << "-- prediction :";
		for (int j = 0; j < testGroup[i].out.size(); j++) cout << testGroup[i].out[j] << "\t";
		cout << endl;
	}

	system("pause");
	return 0;
}