import { ChatWindowMessage } from "@/schema/ChatWindowMessage";

import { Voy as VoyClient } from "voy-search";

import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { WebPDFLoader } from "langchain/document_loaders/web/pdf";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { VoyVectorStore } from "@langchain/community/vectorstores/voy";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from "@langchain/core/prompts";
import { RunnableSequence, RunnablePick } from "@langchain/core/runnables";
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

/* mistral ou mistral:instruct semble le mieux ; all-minilm:l6-v2 moyen. initial : nomic-embed-text */
const embeddings = new OllamaEmbeddings({
  model: "mistral:instruct", /* llama2:13b ou 7b OK */
  baseUrl: "http://localhost:11434",
});

const voyClient = new VoyClient();
const vectorstore = new VoyVectorStore(voyClient, embeddings);

/*const ollama = new ChatOllama({
  baseUrl: "http://localhost:11435",
  temperature: 0.3,
  model: "mistral",
});*/

/* mistral : pas convainquant... zephyr non plus, mistral+mistral semble le meilleur ; llama2:7b ou 13b OK ! */
const ollama = new ChatOllama({
  baseUrl: "http://localhost:11435",
  temperature: 0.1,
  model: "mistral",
});

const RESPONSE_SYSTEM_TEMPLATE = `Tu es un chercheur expérimenté, expert dans l'interprétation et la réponse à des questions posées sur des sources données. En utilisant le contexte fourni, répond en français aux questions de l'utilisateur du mieux que tu peux, en utilisant les ressources fournies.
Génère une réponse concise pour une question donnée, basée sur les résultats de la recherche (URL et contenu), en français. Tu dois utiliser uniquement l'information fournie par les résultats de recherche. Prends un ton journalistique, non biaisé. Combine les résultats ensemble dans une réponse cohérente. Ne répète pas de texte. Réponds toujours en français.
S'il n'y a rien de pertinent dans le contexte par rapport à la question posée, réponds juste "Hmm, je ne suis pas sûr." N'essaie pas d'inventer une réponse.
Tout ce qui suit entre les blocs html \`context\` vient d'une banque de connaissances, et ne fait pas partie de la conversation avec l'utilisateur.
<context>
    {context}
<context/>

RAPPELLE-TOI : S'il n'y a rien de pertinent dans le contexte, réponds juste "Hmm, je ne suis pas sûr." N'essaie pas d'inventer une réponse. Tout ce qui est entre les blocs 'context' précédents vient d'une base de connaissances, et ne fait pas partie de la conversation avec l'utilisateur. Réponds en français.`;

const responseChainPrompt = ChatPromptTemplate.fromMessages<{
  context: string;
  chat_history: BaseMessage[];
  question: string;
}>([
  ["system", RESPONSE_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("chat_history"),
  ["user", `{input}`],
]);

const embedPDF = async (pdfBlob: Blob) => {
  const pdfLoader = new WebPDFLoader(pdfBlob, { parsedItemSeparator: " " });
  const docs = await pdfLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2048, /* orig: 500 */
  chunkOverlap: 50, /* orig: 50 */
});

  const splitDocs = await splitter.splitDocuments(docs);

  self.postMessage({
    type: "log",
    data: splitDocs,
  });

  await vectorstore.addDocuments(splitDocs);
};

const _formatChatHistoryAsMessages = async (
  chatHistory: ChatWindowMessage[],
) => {
  return chatHistory.map((chatMessage) => {
    if (chatMessage.role === "human") {
      return new HumanMessage(chatMessage.content);
    } else {
      return new AIMessage(chatMessage.content);
    }
  });
};

const queryVectorStore = async (messages: ChatWindowMessage[]) => {
  const text = messages[messages.length - 1].content;
  const chatHistory = await _formatChatHistoryAsMessages(messages.slice(0, -1));

  const documentChain = await createStuffDocumentsChain({
    llm: ollama,
    prompt: responseChainPrompt,
    documentPrompt: PromptTemplate.fromTemplate(
      `<doc>\n{page_content}\n</doc>`,
    ),
  });

  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      // "Given the above conversation, generate a natural language search query to look up in order to get information relevant to the conversation. Do not respond with anything except the query.",
      "Partant de la conversation au-dessus, génère une requête en langage naturel pour chercher, afin de trouver l'information pertinente pour la conversation. Ne réponds avec rien sauf la requête.",
    ],
  ]);

  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: ollama,
    retriever: vectorstore.asRetriever(),
    rephrasePrompt: historyAwarePrompt,
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever: historyAwareRetrieverChain,
  });

  const fullChain = RunnableSequence.from([
    retrievalChain,
    new RunnablePick("answer"),
  ]);

  const stream = await fullChain.stream({
    input: text,
    chat_history: chatHistory,
  });

  for await (const chunk of stream) {
    if (chunk) {
      self.postMessage({
        type: "chunk",
        data: chunk,
      });
    }
  }

  self.postMessage({
    type: "complete",
    data: "OK",
  });
};

// Listen for messages from the main thread
self.addEventListener("message", async (event: any) => {
  self.postMessage({
    type: "log",
    data: `Received data!`,
  });

  if (event.data.pdf) {
    try {
      await embedPDF(event.data.pdf);
    } catch (e: any) {
      self.postMessage({
        type: "error",
        error: e.message,
      });
      throw e;
    }
  } else {
    try {
      await queryVectorStore(event.data.messages);
    } catch (e: any) {
      self.postMessage({
        type: "error",
        error: `${e.message}. Make sure you are running Ollama.`,
      });
      throw e;
    }
  }

  self.postMessage({
    type: "complete",
    data: "OK",
  });
});
