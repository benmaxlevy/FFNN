using System.Net.Mail;
using System.Runtime.CompilerServices;

namespace FeedForwardNeuralNetwork;

public class NeuralNetwork
{
    private List<List<double[]>> inputs;
    private double[] theta;
    private List<double[]> outputs;

    public NeuralNetwork()
    {
        
    }
    
    private double Activation(double[] input)
    {
        double output = 0.0;
        
        foreach(double x in input)
        {
            foreach (double w in theta)
            {
                output += x*w;
            }
        }

        return output;
    }

    public double Cost()
    {
        double cost = 0.0;
        
        for(int i = 0; i < inputs.Count; i++)
            for(int j = 0; j < inputs[i].Count; j++)
            // (summed) squared cost
            cost += Math.Pow(Activation(inputs[i][j]) - outputs[i][j], 2);
        
        // multiply the cost by 1/2 to have ∂/∂theta_j not have a 2 in front (makes it a bit easier to differentiate/less verbose code)
        return 0.5*cost;
    }

    public double CostDerivative(int featureIndex)
    {
        double derivative = 0.0;

        for (int i = 0; i < inputs.Count; i++)
            for (int j = 0; j < inputs[i].Count; j++)
            // add derivative of cost func to output var
            derivative += (inputs[i][j][featureIndex] * theta[featureIndex]) * inputs[i][j][featureIndex] -
                          (inputs[i][j][featureIndex] * outputs[i][j]);
        // find average of derivatives
        return derivative / inputs.Count;
    }
}