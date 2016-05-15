using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Project
{
    class BackPropagation
    {
        private int HiddenLayers_Num, Out_Neurn;
        private List<int> HidLayerPerceptron_Num;
        private List<List<Perceptron>> Arch;
        private double Learing_Rate;

        public BackPropagation(int _HiddenLayers_Num, List<int> _HidLayerPerceptron_Num, int _Out_Neurn , int Num_Feature , double _Learing_Rate = 0.01)
        {
            // Our Architecture :-
            // Arch is the Whole Architecture (Consist of Many Layers)
            // Arch[i] is the Current Layer
            // Arch[i][j] is the Current Perceptron in the Layer i
            // Arch[i][j] is A Perceptron Contains its Input (Weights), Activation Function (Sigmoid) , Error , Adder, Activation Derivative ,.....
            // Please Take A Fast Look A Perceptron Class :D

            // hint add Out_Neurn In the Hidden Layers to Loop in All Layer at Once but it suppose not to do that but it is To reduce time and code
            _HidLayerPerceptron_Num.Add(_Out_Neurn);
            HidLayerPerceptron_Num = _HidLayerPerceptron_Num;

            HiddenLayers_Num = _HiddenLayers_Num;
            Out_Neurn = _Out_Neurn;
            Learing_Rate = _Learing_Rate;

            Arch = new List<List<Perceptron>>();

            // ----------------------------------------------------Built Our Architecture -------------------------------------------------------------

            // Loop For HiddenLayers + Output Layer = Hidden Layers + 1     (True :D)
            for (int i = 0; i < HiddenLayers_Num + 1; i++)
            {
                // Every Layer Consist of List of Perceptrons
                Arch.Add(new List<Perceptron>());

                // Loop in This List of Perceptrons
                for (int j = 0; j < HidLayerPerceptron_Num[i]; j++)
                {
                    // Determine the Weights of the Perceptrons for the current layer

                    // First Layer Perceptrons have (128 X as Input (Descriptors))
                    if (i == 0)
                        Arch[i].Add(new Perceptron(Num_Feature));
                    // Other Layers Perceptrons have (n Weights = Number of Perceptrons in the previous Layer)
                    else
                        Arch[i].Add(new Perceptron(Arch[i - 1].Count));
                }
            }
        }

        private void Feed_Forward(List<double> Data)
        {
            List<double> InputData;

            // NN Layers Loop
            for (int i = 0; i < Arch.Count; i++)
            {
                // 1- Determine Input for the Current Layer :

                // for First Layer : The Input is the 128 X
                if (i == 0)
                {
                    InputData = Data;
                }
                // for Any Other Layer : The Input is the Activation Results from the previous Layer
                else
                {
                    InputData = new List<double>(Arch[i - 1].Count);
                    for (int k = 0; k < Arch[i - 1].Count; k++)
                        InputData.Add(Arch[i - 1][k].get_ActiveResult());
                }
                // 2- Loop in All Perceptron in the Current Layer
                for (int j = 0; j < Arch[i].Count; j++)
                {
                    // Calculate Sum (X * W) using Adder
                    Arch[i][j].Adder(InputData);
                    // Calculate The Activation Result in the Preceptron
                    Arch[i][j].ActivationFunction();
                }
            }
            // End FeedForword
        }

        private void Back_Word(int Desired, List<double> Data)
        {
            int size = Arch[Arch.Count - 1].Count;
            List<double> DesiredOut = new List<double>(size);
            List<double> Weights;
            double Error;

            // Built a DesiredOut List
            for (int i = 0; i < size; i++)
                if (i == Desired)
                    DesiredOut.Add(1);
                else
                    DesiredOut.Add(0);

            // Calculate Error for The Output Layer -------> (D - A) * (DerivativeSigmoid)
            for (int i = 0; i < size; i++)
                Arch[Arch.Count - 1][i].set_error((DesiredOut[i] - Arch[Arch.Count - 1][i].get_ActiveResult()) * Arch[Arch.Count - 1][i].DerivativeSigmoid());

            // Calculate Error for the Others Layer -----> (DerivativeSigmoid) * (Summation of (Error * W))
            // Please don't be Confused in this 2 loop So Concentrate And Draw An Example With Your Hand

            // Loop For Layers
            for (int i = Arch.Count - 2; i >= 0; i--)
            {
                // Loop For Perceptron in the Current Layers
                for (int j = 0; j < Arch[i].Count; j++)
                {
                    Error = 0;
                    // Get All Weigths and the Error from the Layer After Me And Summation All (W * E)
                    for (int k = 0; k < Arch[i + 1].Count; k++)
                    {
                        Weights = new List<double>((Arch[i + 1][k]).get_weights());
                        Error += (Arch[i + 1][k].get_error() * Weights[j]);
                    }
                    // Set The Error for The Current Perceptron (All Error * Derivative Sigmoid)
                    Arch[i][j].set_error(Error * Arch[i][j].DerivativeSigmoid());
                }
            }
            // End Back_Word
        }

        private void UpdateWeights(List<double> Data)
        {
            // To Updata Weights We Go Forword A Gain :D
            List<double> InputData;
            List<double> Weights;
            List<double> New_Weights;
            double New_Bais;

            // The Same as Forword Function ()
            for (int i = 0; i < Arch.Count; i++)
            {
                // Preper Input Data Depent in The Current Layer
                if (i == 0)
                    InputData = Data;
                else
                {
                    InputData = new List<double>(Arch[i - 1].Count);
                    for (int k = 0; k < Arch[i - 1].Count; k++)
                        InputData.Add(Arch[i - 1][k].get_ActiveResult());
                }
                // Loop For Perceptrons in the Current Layer, Update Weigths (New W = Old W + (Learing Rate * Percepron Error * Input For This Perceptron))
                for (int j = 0; j < Arch[i].Count; j++)
                {
                    Weights = new List<double>(Arch[i][j].get_weights());
                    New_Weights = new List<double>(Weights.Count);

                    // get the New Wigths
                    for (int k = 0; k < Weights.Count; k++)
                        New_Weights.Add(Weights[k] + (Learing_Rate * Arch[i][j].get_error() * Data[k]));
                    Arch[i][j].set_weights(New_Weights);

                    // get the new Bais
                    New_Bais = Arch[i][j].get_bais() + (Learing_Rate * Arch[i][j].get_error() * 1);
                    Arch[i][j].set_bais(New_Bais);
                }
            }
            // End UpdateWeights
        }

        public void Training(List<KeyValuePair<int, List<double>>> Data, int Num_Epoch)
        {
            // Look in Data :-
            // Data Consist of Group of Descriptors and Its Desired Output (It's Class)
            // Every Descriptors Consist of 128 Value (X) The Input For NN

            // Loop For Epoch in Every Picture Will Send and This Will Discuss Later (May Change)
            for (int i = 0; i < Num_Epoch; i++)
            {
                // Loop for Data
                for (int Index = 0; Index < Data.Count; Index++)
                {
                    // Go Forward (Calculate Activation Result for Every Perceptron and The Out of NN)
                    Feed_Forward(Data[Index].Value);

                    // Go Backword (Calculate the Error for Every Perceptron)
                    Back_Word(Data[Index].Key, Data[Index].Value);

                    // UpdateWeights for Every Weights in the NN
                    UpdateWeights(Data[Index].Value);
                }
            }
            // End Of Training
        }

        public List<int> Testing(List<KeyValuePair<int, List<double>>> Data)
        {
            // hint Meaning Of Winner is The Winner Perceptron in the Output Layer
            int WinnerIndex;
            double WinnerValue;

            // Out List Is The Actual Output From The Output Layer
            List<double> Out = new List<double>(Arch[Arch.Count - 1].Count);

            // Classes List Is The Counter for Winner Classes
            // Example : {0 , 3 , 1 , 0 , 0} that means Class 1 has been won 3 times and Class 2 has been won 1 time and the Other have been won 0 time
            List<int> Classes = new List<int>(Arch[Arch.Count - 1].Count);

            // All Classes Counter = 0
            for (int i = 0; i < Arch[Arch.Count - 1].Count; i++)
                Classes.Add(0);

            // Loop in The Date (Descritors)
            for (int Index = 0; Index < Data.Count; Index++)
            {
                // Go Forword
                Feed_Forward(Data[Index].Value);
                // Get The Output Layer Result
                Out.Clear();
                for (int i = 0; i < Arch[Arch.Count - 1].Count; i++)
                    Out.Add(Arch[Arch.Count - 1][i].get_ActiveResult());

                // Get The Winner form The Output layer Perceptron and then ++ it in Classes List
                WinnerIndex = 0;
                WinnerValue = Out[0];
                for (int i = 1; i < Out.Count; i++)
                {
                    if (WinnerValue < Out[i])
                    {
                        WinnerIndex = i;
                        WinnerValue = Out[i];
                    }
                }
                Classes[WinnerIndex]++;
            }

            // if Classes[i] >= 3 that means this Class Alread Exist in the Picture :D Mohamed told me that and some body told mohamed that :D
            // Save All Winner Classes in WinnerObjects List
            List<int> WinnerObjects = new List<int>();
            for (int i = 0; i < Classes.Count; i++)
                if (Classes[i] >= 3)
                    WinnerObjects.Add(i);

            // Reture WinnerObjects
            return WinnerObjects;

            // End Testing
        }
    }
}
