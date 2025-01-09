using DotNetEnv;

class Program
{
    static async Task Main(string[] args)
    {
        // Configuration
        string modelName = "sentence-transformers/all-MiniLM-L6-v2";
        Env.Load();
        string? apiKey = Environment.GetEnvironmentVariable("HUGGING_FACE_API_KEY");
        
        if (string.IsNullOrEmpty(apiKey))
        {
            throw new Exception("HUGGING_FACE_API_KEY is not set in the environment.");
        }

        // Predefined list of sentences
        var texts = new List<string>
        {
            "Testing method that generates embeddings for a list of input texts.",
            "Testing TorchSharp to check if CUDA is available.",
            "Using Microsoft Semantic Kernel."
        };

        Console.WriteLine("Generating embeddings for predefined texts:");
        foreach (var text in texts)
        {
            Console.WriteLine($"- {text}");
        }

        try
        {
            var embeddings = await EmbeddingGenerator.GenerateEmbeddingsStaticAsync(texts, modelName, apiKey);

            Console.WriteLine("\nGenerated embeddings:");
            for (int i = 0; i < texts.Count; i++)
            {
                Console.WriteLine($"\nText: {texts[i]}");
                Console.WriteLine($"Embedding ({embeddings[i].Count} dimensions): [{string.Join(", ", embeddings[i].Take(5))}...]");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}