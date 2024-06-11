using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Threading.Tasks;

namespace MLLTesterCMD.DigitClassification
{
    public static class ImageConverter
    {
        public static Bitmap ToBitmap(DigitData data)
        {
            Bitmap bmp = new Bitmap(data.Width, data.Height);
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                    bmp.SetPixel(x, y, Color.FromArgb(data.Data[y * data.Width + x], data.Data[y * data.Width + x], data.Data[y * data.Width + x]));
            return bmp;
        }

        public static DigitData ToDigitData(Bitmap bmp)
        {
            DigitData data = new DigitData();
            data.Width = bmp.Width;
            data.Height = bmp.Height;
            data.Data = new byte[data.Width * data.Height];
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    Color c = bmp.GetPixel(x, y);
                    data.Data[y * data.Width + x] = (byte)(int)((c.R + c.G + c.B) / 3.0);
                }
            return data;
        }
    }
}
