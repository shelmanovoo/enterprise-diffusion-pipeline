using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;

namespace Enterprise3DPipeline
{
    class Program
    {
        private static readonly HttpClient http = new HttpClient
        {
            Timeout = TimeSpan.FromMinutes(10)
        };

        private static readonly string ollamaUrl =
            "http://localhost:11434/v1/chat/completions";

        private static readonly string diffusionUrl =
            "http://192.168.1.37:5001/generate";

        static async Task Main()
        {
            Console.WriteLine("=== Enterprise Diffusion Pipeline ===");

            string request =
                "Создать промышленного робота, high detail, CAD style";

            var prompt = await GeneratePromptOllama(request);

            Console.WriteLine("\nGenerated prompt:");
            Console.WriteLine(prompt);

            var imagePath = await RunDiffusion(prompt);

            Console.WriteLine($"\nImage saved: {imagePath}");
        }

        static async Task<string> GeneratePromptOllama(string userRequest)
        {
            var payload = new
            {
                model = "llama3",
                messages = new[]
                {
                    new {
                        role = "system",
                        content =
                        "You are an expert in diffusion prompting. " +
                        "Generate highly detailed SDXL prompts."
                    },
                    new {
                        role = "user",
                        content = userRequest
                    }
                },
                temperature = 0.7
            };

            var response =
                await http.PostAsJsonAsync(ollamaUrl, payload);

            response.EnsureSuccessStatusCode();

            var json =
                await response.Content.ReadFromJsonAsync<JsonElement>();

            return json
                .GetProperty("choices")[0]
                .GetProperty("message")
                .GetProperty("content")
                .GetString();
        }

        static async Task<string> RunDiffusion(string prompt)
        {
            var payload = new
            {
                prompt = prompt,
                width = 1024,
                height = 1024,
                steps = 30
            };

            var response =
                await http.PostAsJsonAsync(diffusionUrl, payload);

            response.EnsureSuccessStatusCode();

            var imageBytes =
                await response.Content.ReadAsByteArrayAsync();

            var path = $"output_{DateTime.Now.Ticks}.png";

            await System.IO.File.WriteAllBytesAsync(path, imageBytes);

            return path;
        }
    }
}