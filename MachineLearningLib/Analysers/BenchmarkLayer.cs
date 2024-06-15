using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.Analysers
{
    public class BenchmarkLayer : Layer
    {
        public int CalculateCounter { get; private set; } = 0;
        long CalculateEntireTime = 0;
        public long MaxCalculateDurationMillis { get; private set; } = -1;
        public long MinCalculateDurationMillis { get; private set; } = long.MaxValue;
        public long AverageCalculateDurationMillis { get { return CalculateEntireTime / CalculateCounter; } }

        public int TrainCounter { get; private set; } = 0;
        long TrainEntireTime = 0;
        public long MaxTrainDurationMillis { get; private set; } = -1;
        public long MinTrainDurationMillis { get; private set; } = long.MaxValue;
        public long AverageTrainDurationMillis { get { return TrainEntireTime / TrainCounter; } }

        public int SaveCounter { get; private set; } = 0;
        long SaveEntireTime = 0;
        public long MaxSaveDurationMillis { get; private set; } = -1;
        public long MinSaveDurationMillis { get; private set; } = long.MaxValue;
        public long AverageSaveDurationMillis { get { return SaveEntireTime / SaveCounter; } }

        public int LoadCounter { get; private set; } = 0;
        long LoadEntireTime = 0;
        public long MaxLoadDurationMillis { get; private set; } = -1;
        public long MinLoadDurationMillis { get; private set; } = long.MaxValue;
        public long AverageLoadDurationMillis { get { return LoadEntireTime / LoadCounter; } }

        public long WeightInitializationDuration { get; private set; }

        public BenchmarkLayer(int neurons) : base(neurons)
        {
            
        }

        public override void InitFromPreviousLayer()
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            base.InitFromPreviousLayer();
            sw.Stop();
            WeightInitializationDuration = sw.ElapsedMilliseconds;
        }

        public override void Calculate()
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            base.Calculate();
            sw.Stop();
            CalculateCounter++;
            CalculateEntireTime += sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds > MaxCalculateDurationMillis)
                MaxCalculateDurationMillis = sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds < MinCalculateDurationMillis)
                MinCalculateDurationMillis = sw.ElapsedMilliseconds;
        }

        public override void Train(float learningRate)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            base.Train(learningRate);
            sw.Stop();
            TrainCounter++;
            TrainEntireTime += sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds > MaxTrainDurationMillis)
                MaxTrainDurationMillis = sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds < MinTrainDurationMillis)
                MinTrainDurationMillis = sw.ElapsedMilliseconds;
        }

        public override void Save(BinaryWriter bw)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            base.Save(bw);
            sw.Stop();
            SaveCounter++;
            SaveEntireTime += sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds > MaxSaveDurationMillis)
                MaxSaveDurationMillis = sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds < MinSaveDurationMillis)
                MinSaveDurationMillis = sw.ElapsedMilliseconds;
        }

        public override void Load(BinaryReader br)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            base.Load(br);
            sw.Stop();
            LoadCounter++;
            LoadEntireTime += sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds > MaxLoadDurationMillis)
                MaxLoadDurationMillis = sw.ElapsedMilliseconds;
            if (sw.ElapsedMilliseconds < MinLoadDurationMillis)
                MinLoadDurationMillis = sw.ElapsedMilliseconds;
        }

        public void ResetCalculate()
        {
            CalculateCounter = 0;
            CalculateEntireTime = 0;
            MaxCalculateDurationMillis = -1;
            MinCalculateDurationMillis = long.MaxValue;
        }

        public void ResetTrain()
        {
            TrainCounter = 0;
            TrainEntireTime = 0;
            MaxTrainDurationMillis = -1;
            MinTrainDurationMillis = long.MaxValue;
        }

        public void ResetSave()
        {
            SaveCounter = 0;
            SaveEntireTime = 0;
            MaxSaveDurationMillis = -1;
            MinSaveDurationMillis = long.MaxValue;
        }

        public void ResetLoad()
        {
            LoadCounter = 0;
            LoadEntireTime = 0;
            MaxLoadDurationMillis = -1;
            MinLoadDurationMillis = long.MaxValue;
        }

        public void ResetAll()
        {
            ResetCalculate();
            ResetTrain();
            ResetSave();
            ResetLoad();
        }
    }
}
