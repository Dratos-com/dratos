'use client'
import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { Button } from "@/components/ui/button";
import { SideBySideChatViewer } from "@/components/side-by-side-chat-viewer";
import { fetchConversationPage, createBranch, mergeBranches, sendMessage, editMessage, fetchBranches } from '@/lib/api';
interface Conversation {
    id: string;
    commits: any[];
    branches: any[];
    forks: any[];
    fractals?: any;
    messages: {
      id: string;
      content: string;
      sender: 'user' | 'ai';
      timestamp: string;
    }[];
  }
  
  const initialConversation: Conversation = {
    id: 'initial',
    commits: [
      {
        id: 'initial-commit',
        message: 'Welcome to the conversation!',
        author: 'System',
        timestamp: new Date().toISOString(),
        changes: {
          added: ['Hello! Welcome to our conversation. How may I assist you today?'],
          removed: []
        }
      }
    ],
    branches: [],
    forks: [],
    fractals: {
      id: 'root',
      content: '',
      children: [],
      position: [0, 0, 0],
      color: ''
    },
    messages: [
      {
        id: 'initial-message',
        content: 'Hello! Welcome to our conversation. How may I assist you today?',
        sender: 'ai',
        timestamp: new Date().toISOString()
      }
    ]
  };
export default function SideBySideChatPage() {
    const [conversations, setConversations] = useState<{ [key: string]: Conversation }>({
        initial: initialConversation,
      });
      const [activeConversations, setActiveConversations] = useState<string[]>(['initial']);
      const [branches, setBranches] = useState<any[]>([]);
      const [error, setError] = useState<string | null>(null);
    
      useEffect(() => {
        loadConversation();
        loadBranches();
      }, []);
    
      const loadConversation = async () => {
        try {
          const conversationData = await fetchConversationPage('initial', 1, 20);
          setConversations(prev => ({
            ...prev,
            initial: {
              ...prev.initial,
              messages: conversationData.messages,
            },
          }));
        } catch (error) {
          console.error('Error loading conversation:', error);
          setError('Failed to load conversation');
        }
      };
    
      const loadBranches = async () => {
        try {
          console.log('Fetching branches...');
          const branchesData = await fetchBranches('initial');
          console.log('Branches data received:', branchesData);
          if (Array.isArray(branchesData) && branchesData.length > 0) {
            setBranches(branchesData);
          } else {
            console.log('No branches found or empty array returned');
            setBranches([]);
          }
        } catch (error) {
          console.error('Error loading branches:', error);
          setError('Failed to load branches');
          setBranches([]);
        }
      };
    
      const handleCreateBranch = async (conversationId: string, newBranchId: string) => {
        try {
          console.log('Creating new branch:', newBranchId, 'for conversation:', conversationId);
          const latestCommit = conversations[conversationId]?.commits?.length > 0
            ? conversations[conversationId].commits[conversations[conversationId].commits.length - 1]
            : { id: 'initial' };

          await createBranch(conversationId, newBranchId, latestCommit.id);
          console.log('Branch created successfully');
          await loadBranches();
          
          // Initialize the new conversation branch
          setConversations(prev => ({
            ...prev,
            [newBranchId]: {
              id: newBranchId,
              commits: [],
              branches: [],
              forks: [],
              messages: [],
            },
          }));
          setActiveConversations(prev => [...prev, newBranchId]);
        } catch (error) {
          console.error('Error creating branch:', error);
          setError('Failed to create branch');
        }
      };
    
      const handleMergeBranches = async (sourceBranch: string, targetBranch: string) => {
        try {
          await mergeBranches(conversations.initial.id, sourceBranch, targetBranch);
          await loadConversation();
        } catch (error) {
          console.error('Error merging branches:', error);
          setError('Failed to merge branches');
        }
      };
    
      const handleSendMessage = async (conversationId: string, content: string, sender: 'user' | 'ai') => {
        try {
          const newMessage = await sendMessage(conversationId, content, sender);
          setConversations(prev => ({
            ...prev,
            [conversationId]: {
              ...prev[conversationId],
              messages: [...prev[conversationId].messages, newMessage],
              commits: [...prev[conversationId].commits, { id: newMessage.commit_id, messages: newMessage.history }],
            },
          }));
        } catch (error) {
          console.error('Error sending message:', error);
          setError('Failed to send message');
        }
      };
    
      const handleEditMessage = async (conversationId: string, messageId: string, newContent: string) => {
        try {
          const updatedMessage = await editMessage(conversationId, messageId, newContent);
          setConversations(prev => ({
            ...prev,
            [conversationId]: {
              ...prev[conversationId],
              messages: prev[conversationId].messages.map(msg =>
                msg.id === messageId ? updatedMessage : msg
              ),
            },
          }));
        } catch (error) {
          console.error('Error editing message:', error);
          setError('Failed to edit message');
        }
      };
    
      const loadMoreCommits = () => {
        // Implement load more commits logic
      };
    
      const refreshConversation = async () => {
        // Implement refresh conversation logic
      };
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-4xl font-bold mb-8">Memento Chat</h1>
      <SideBySideChatViewer
        conversations={conversations}
        activeConversations={activeConversations}
        onCreateBranch={handleCreateBranch}
        onMergeBranches={handleMergeBranches}
        onSendMessage={handleSendMessage}
        onEditMessage={handleEditMessage}
        branches={branches}
      />
      {error && <div className="text-red-500 mt-4">{error}</div>}
    </main>
  );
}
