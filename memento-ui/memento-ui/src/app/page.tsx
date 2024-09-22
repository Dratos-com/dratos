'use client'
import React, { useState, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import MementoFractals from "@/components/memento-fractals-3d";
import { BentoConversationUi } from "@/components/bento-conversation-ui";
import { EnhancedTimePortalConversations } from "@/components/enhanced-time-portal-conversations";
import { SideBySideChatViewer } from "@/components/side-by-side-chat-viewer";
import { TimePortalConversations } from "@/components/time-portal-conversations";
import { fetchConversationPage, createBranch, mergeBranches, sendMessage, editMessage, fetchBranches } from '@/lib/api';
import { Button } from "@/components/ui/button";

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

export default function Home() {
  const [conversation, setConversation] = useState<Conversation>(initialConversation);
  const [currentCommitIndex, setCurrentCommitIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [branches, setBranches] = useState<string[]>([]);
  const [activeConversations, setActiveConversations] = useState<[string, string]>(['initial', '']);

  const loadConversation = async () => {
    try {
      setLoading(true);
      const conversationData = await fetchConversationPage('initial', page, pageSize);
      if (conversationData && conversationData.commits && conversationData.commits.length > 0) {
        setConversation(prevConversation => {
          const newMessages = conversationData.commits.map(commit => ({
            id: commit.id,
            content: commit.message,
            sender: commit.author === 'User' ? 'user' : 'ai',
            timestamp: commit.timestamp
          }));
          
          return {
            ...(prevConversation || {}),
            ...conversationData,
            commits: [...(prevConversation?.commits || []), ...conversationData.commits],
            messages: [...(prevConversation?.messages || []), ...newMessages],
          } as Conversation;
        });
      } else {
        setConversation(initialConversation);
      }
      setLoading(false);
    } catch (err) {
      console.error('Failed to load conversation:', err);
      setError('Failed to load conversation');
      setConversation(initialConversation);
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConversation();
  }, [page, pageSize]);

  const handleTimeTravel = (direction: 'back' | 'forward') => {
    setCurrentCommitIndex(prevIndex => {
      if (direction === 'back' && prevIndex > 0) {
        return prevIndex - 1;
      } else if (direction === 'forward' && conversation && prevIndex < conversation.commits.length - 1) {
        return prevIndex + 1;
      }
      return prevIndex;
    });
  };

  const handleCreateBranch = async (conversationId: string, newBranchId: string) => {
    try {
      const latestCommitId = conversation.commits[conversation.commits.length - 1]?.id || 'initial';
      const result = await createBranch(conversationId, newBranchId, latestCommitId);
      
      // Update the branches state with the new branch
      setBranches(prevBranches => [...prevBranches, result.branch_name]);
      
      // Update active conversations to show the new branch
      setActiveConversations([conversationId, result.branch_name]);

      // Update the conversation state with the new branch and commit
      setConversation(prevConversation => ({
        ...prevConversation,
        branches: [...prevConversation.branches, result.branch_name],
        commits: [...prevConversation.commits, { id: result.commit_id, messages: result.history }]
      }));

      // Optionally, you can also refresh the conversation data here
      await loadConversation();
    } catch (error) {
      console.error('Error creating branch:', error);
      setError('Failed to create branch');
    }
  };

  const handleMergeBranches = async (sourceBranch: string, targetBranch: string) => {
    try {
      await mergeBranches(conversation.id, sourceBranch, targetBranch);
      await loadConversation();
    } catch (error) {
      console.error('Error merging branches:', error);
      setError('Failed to merge branches');
    }
  };

  const handleSendMessage = async (conversationId: string, content: string, sender: 'user' | 'ai') => {
    try {
      const newMessage: MessageResponse = await sendMessage(conversationId, content, sender);
      setConversation(prevConversation => ({
        ...prevConversation,
        messages: [...prevConversation.messages, newMessage],
        commits: [...prevConversation.commits, { id: newMessage.commit_id, messages: newMessage.history }]
      }));
    } catch (error) {
      console.error('Error sending message:', error);
      setError('Failed to send message');
    }
  };

  const handleEditMessage = async (conversationId: string, messageId: string, newContent: string) => {
    try {
      const updatedMessage = await editMessage(conversationId, messageId, newContent);
      setConversation(prevConversation => ({
        ...prevConversation,
        messages: prevConversation.messages.map(msg =>
          msg.id === messageId ? updatedMessage : msg
        )
      }));
    } catch (error) {
      console.error('Error editing message:', error);
      setError('Failed to edit message');
    }
  };

  const loadMoreCommits = () => {
    setPage(prevPage => prevPage + 1);
  };

  const refreshConversation = async () => {
    setPage(1);
    await loadConversation();
  };

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <h1>Memento UI</h1>
      {conversation && (
        <TimePortalConversations 
          conversationId={conversation.id}
          conversation={conversation}
          currentCommitIndex={currentCommitIndex}
          onTimeTravel={handleTimeTravel}
          onRefresh={refreshConversation}
        />
      )}
      <MementoFractals conversationHistory={conversation?.fractals || initialConversation.fractals} />
      <BentoConversationUi />
      <EnhancedTimePortalConversations />
      <SideBySideChatViewer
        conversations={[conversation]}
        activeConversations={activeConversations}
        onCreateBranch={handleCreateBranch}
        onMergeBranches={handleMergeBranches}
        onSendMessage={handleSendMessage}
        onEditMessage={handleEditMessage}
        branches={branches}
      />
      {conversation && conversation.commits.length % pageSize === 0 && (
        <Button onClick={loadMoreCommits}>Load More</Button>
      )}
    </div>
  );
}
