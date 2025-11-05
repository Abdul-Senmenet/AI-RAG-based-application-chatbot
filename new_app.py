from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize your RAG agent (same code as before)
class RAGAgent:
    def __init__(self):
        # LLM & Embeddings
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # PDF Loading
        pdf_path = "research1.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load()
        print(f"PDF loaded with {len(pages)} pages")

        # Text Chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pages_split = text_splitter.split_documents(pages)

        # Chroma Vector Store
        persist_directory = r"C:\Users\saabd\Desktop\langGraph"
        collection_name = "research_stuff"

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        print("ChromaDB vector store created!")

        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Setup tools and graph
        self.setup_agent()

    @tool
    def retriever_tool(self, query: str) -> str:
        """
        This tool searches and returns information from the research paper PDF.
        """
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found in the research paper."
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    def setup_agent(self):
        tools = [self.retriever_tool]
        self.tools_dict = {t.name: t for t in tools}

        # Agent State
        class AgentState(TypedDict):
            messages: Sequence[BaseMessage]

        def should_continue(state: AgentState):
            last_message = state['messages'][-1]
            return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

        # LLM Call
        system_prompt = """
        You are an intelligent AI assistant who answers questions about indivisible fair division and related topics.
        Use the retriever tool to look up information from the research paper.
        Always cite specific parts of the document in your answers.
        """

        def call_llm(state: AgentState) -> AgentState:
            messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
            message = self.llm.invoke(messages)
            return {'messages': [message]}

        # Tool Execution
        def take_action(state: AgentState) -> AgentState:
            tool_calls = state['messages'][-1].tool_calls
            results = []
            for t in tool_calls:
                if t['name'] not in self.tools_dict:
                    result = "Incorrect Tool Name, please use the available tool."
                else:
                    result = self.tools_dict[t['name']].invoke(t['args'].get('query', ''))
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            return {'messages': results}

        # State Graph
        graph = StateGraph(AgentState)
        graph.add_node("llm", call_llm)
        graph.add_node("retriever_agent", take_action)
        graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
        graph.add_edge("retriever_agent", "llm")
        graph.set_entry_point("llm")
        self.rag_agent = graph.compile()

    def ask_question(self, question: str) -> str:
        """Ask a question to the RAG agent and return the response"""
        try:
            messages = [HumanMessage(content=question)]
            result = self.rag_agent.invoke({"messages": messages})
            return result['messages'][-1].content
        except Exception as e:
            print(f"Error in RAG agent: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."

# Initialize RAG agent
print("Initializing RAG Agent...")
rag_agent = RAGAgent()
print("RAG Agent initialized successfully!")

# HTML template for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Agent Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 20px 20px 0 0;
        }

        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .chat-header p {
            font-size: 14px;
            opacity: 0.9;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.4;
            animation: slideIn 0.3s ease-out;
            white-space: pre-wrap;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .bot-message {
            align-self: flex-start;
            background: #f3f4f6;
            color: #374151;
            border: 1px solid #e5e7eb;
            border-bottom-left-radius: 4px;
        }

        .typing-indicator {
            align-self: flex-start;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 12px 16px;
            display: none;
        }

        .typing-dots {
            display: inline-flex;
            gap: 4px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: #9ca3af;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }

        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e5e7eb;
        }

        .chat-input-form {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            font-size: 14px;
            resize: none;
            min-height: 44px;
            max-height: 120px;
            font-family: inherit;
            transition: border-color 0.3s ease;
        }

        .chat-input:focus {
            outline: none;
            border-color: #4f46e5;
        }

        .send-button {
            padding: 12px 20px;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            min-width: 80px;
        }

        .send-button:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }

        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .welcome-message {
            text-align: center;
            color: #6b7280;
            font-style: italic;
            margin: 40px 0;
        }

        .error-message {
            align-self: center;
            background: #fef2f2;
            color: #dc2626;
            border: 1px solid #fecaca;
            border-radius: 12px;
            padding: 12px 16px;
            font-size: 14px;
            max-width: 90%;
        }

        /* Scrollbar styling */
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 3px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 RAG Agent</h1>
            <p>Ask questions about indivisible fair division research</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                👋 Welcome! I'm here to help you with questions about the research paper. What would you like to know?
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <span style="margin-left: 8px; color: #6b7280;">AI is thinking...</span>
        </div>
        
        <div class="chat-input-container">
            <form class="chat-input-form" id="chatForm">
                <textarea 
                    class="chat-input" 
                    id="chatInput" 
                    placeholder="Type your question here..." 
                    rows="1"
                ></textarea>
                <button type="submit" class="send-button" id="sendButton">
                    Send
                </button>
            </form>
        </div>
    </div>

    <script>
        class RAGChatUI {
            constructor() {
                this.chatMessages = document.getElementById('chatMessages');
                this.chatInput = document.getElementById('chatInput');
                this.sendButton = document.getElementById('sendButton');
                this.chatForm = document.getElementById('chatForm');
                this.typingIndicator = document.getElementById('typingIndicator');
                
                this.initializeEventListeners();
                this.adjustTextareaHeight();
            }

            initializeEventListeners() {
                this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
                
                this.chatInput.addEventListener('input', () => this.adjustTextareaHeight());
                
                this.chatInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.handleSubmit(e);
                    }
                });
            }

            adjustTextareaHeight() {
                this.chatInput.style.height = 'auto';
                this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
            }

            async handleSubmit(e) {
                e.preventDefault();
                
                const question = this.chatInput.value.trim();
                if (!question) return;

                // Add user message
                this.addMessage(question, 'user');
                this.chatInput.value = '';
                this.adjustTextareaHeight();
                
                // Disable input while processing
                this.setInputState(false);
                this.showTypingIndicator();

                try {
                    const response = await this.callRAGAgent(question);
                    this.hideTypingIndicator();
                    this.addMessage(response, 'bot');
                } catch (error) {
                    this.hideTypingIndicator();
                    this.addMessage('Sorry, I encountered an error while processing your question. Please try again.', 'error');
                    console.error('Error calling RAG agent:', error);
                } finally {
                    this.setInputState(true);
                    this.chatInput.focus();
                }
            }

            async callRAGAgent(question) {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                
                const data = await response.json();
                return data.answer;
            }

            addMessage(content, type) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type === 'user' ? 'user-message' : type === 'error' ? 'error-message' : 'bot-message'}`;
                messageDiv.textContent = content;
                
                this.chatMessages.appendChild(messageDiv);
                this.scrollToBottom();
            }

            showTypingIndicator() {
                this.typingIndicator.style.display = 'block';
                this.scrollToBottom();
            }

            hideTypingIndicator() {
                this.typingIndicator.style.display = 'none';
            }

            setInputState(enabled) {
                this.chatInput.disabled = !enabled;
                this.sendButton.disabled = !enabled;
                this.sendButton.textContent = enabled ? 'Send' : 'Sending...';
            }

            scrollToBottom() {
                setTimeout(() => {
                    this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
                }, 100);
            }
        }

        // Initialize the chat UI when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new RAGChatUI();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the chat interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle questions from the chat interface"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"Received question: {question}")
        
        # Get response from RAG agent
        answer = rag_agent.ask_question(question)
        
        print(f"RAG Agent response: {answer}")
        
        return jsonify({
            'answer': answer,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in ask_question endpoint: {e}")
        return jsonify({
            'error': 'An error occurred while processing your question',
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'RAG Agent is running'
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Chat interface will be available at: http://localhost:5000")
    print("API endpoint available at: http://localhost:5000/ask")
    app.run(debug=True, host='0.0.0.0', port=5000)
