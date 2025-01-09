using TorchSharp;
using Microsoft.SemanticKernel.Connectors.HuggingFace;

#pragma warning disable SKEXP0070
#pragma warning disable SKEXP0001

public class EmbeddingGenerator
{
    private readonly HuggingFaceTextEmbeddingGenerationService _embeddingService;

    public EmbeddingGenerator(string model, string apiKey)
    {
        _embeddingService = new HuggingFaceTextEmbeddingGenerationService(
            model: model,
            apiKey: apiKey,
            httpClient: new HttpClient()
        );
    }

    /// <summary>
    /// Static method to generate embeddings for a list of strings using TorchSharp for CUDA acceleration if available.
    /// </summary>
    public static async Task<List<List<float>>> GenerateEmbeddingsStaticAsync(
        List<string> texts,
        string model,
        string apiKey)
    {
        try
        {
            // Check if CUDA is available
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine(torch.cuda.is_available()
                ? "CUDA is available and will be used for embeddings generation."
                : "CUDA is not available, using CPU for embeddings generation.");

            // Create the embedding service
            var embeddingService = new HuggingFaceTextEmbeddingGenerationService(
                model: model,
                apiKey: apiKey,
                httpClient: new HttpClient()
            );

            var embeddingsList = new List<List<float>>();

            // Generate embeddings for each text individually
            foreach (var text in texts)
            {
                var embedding = await embeddingService.GenerateEmbeddingsAsync(new List<string> { text });
                var tensor = torch.tensor(embedding.First().ToArray(), device: device);
                embeddingsList.Add(tensor.to(torch.CPU).data<float>().ToArray().ToList());
            }

            return embeddingsList;
        }
        catch (Exception ex)
        {
            throw new Exception($"Error generating embeddings: {ex.Message}", ex);
        }
    }
}
