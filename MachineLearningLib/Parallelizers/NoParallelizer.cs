using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.Parallelizers
{
    public class NoParallelizer : IParallelizer
    {
        public void Parallelizer(int from, int to, Action<int> action)
        {
            for (int i = from; i < to; i++)
                action(i);
        }
    }
}
