using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.Accelerators
{
    public class NoAccelerator : IAccelerator
    {
        public float[] Add(float[] vec1, float[] vec2)
        {
            float[] res = new float[vec1.Length];
            for(int i = 0; i < vec1.Length; i++)
                res[i] = vec1[i] + vec2[i];
            return res;
        }

        public float[] Add(float a, float[] vec)
        {
            float[] res = new float[vec.Length];
            for (int i = 0; i < vec.Length; i++)
                res[i] = a + vec[i];
            return res;
        }

        public float DotProduct(float[] vec1, float[] vec2)
        {
            float res = 0;
            for (int i = 0; i < vec1.Length; i++)
                res += vec1[i] * vec2[i];
            return res;
        }

        public float DotProductT(float[] vec1, float[][] vec2, int subIndex)
        {
            float res = 0;
            for (int i = 0; i < vec1.Length; i++)
                res += vec1[i] * vec2[i][subIndex];
            return res;
        }

        public float[] Multiply(float[] vec1, float[] vec2)
        {
            float[] res = new float[vec1.Length];
            for (int i = 0; i < vec1.Length; i++)
                res[i] = vec1[i] * vec2[i];
            return res;
        }

        public float[] Multiply(float a, float[] vec)
        {
            float[] res = new float[vec.Length];
            for (int i = 0; i < vec.Length; i++)
                res[i] = a * vec[i];
            return res;
        }
    }
}
