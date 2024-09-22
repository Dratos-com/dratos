'use client'
import React, { useState, useEffect } from 'react';
import MementoFractals from "@/components/memento-fractals-3d";
import { BentoConversationUi } from "@/components/bento-conversation-ui";
import { EnhancedTimePortalConversations } from "@/components/enhanced-time-portal-conversations";
import { SideBySideChatViewer } from "@/components/side-by-side-chat-viewer";
import { TimePortalConversations } from "@/components/time-portal-conversations";
import { fetchConversationPage } from '@/lib/api';
import { Button } from "@/components/ui/button";

interface Conversation {
  id: string;
  commits: any[];
  branches: any[];
  forks: any[];
  fractals?: any;
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
  }
};

export default function Home() {
  const [conversation, setConversation] = useState<Conversation>(initialConversation);
  const [currentCommitIndex, setCurrentCommitIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  const loadConversation = async () => {
    try {
      setLoading(true);
      const conversationData = await fetchConversationPage('initial', page, pageSize);
      if (conversationData && conversationData.commits && conversationData.commits.length > 0) {
        setConversation(prevConversation => ({
          ...(prevConversation || {}),
          ...conversationData,
          commits: [...(prevConversation?.commits || []), ...conversationData.commits],
        }) as Conversation);
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

  const loadMoreCommits = () => {
    setPage(prevPage => prevPage + 1);
  };

  const refreshConversation = async () => {
    setPage(1);
    setConversation(null);
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
      <SideBySideChatViewer />
      {conversation && conversation.commits.length % pageSize === 0 && (
        <Button onClick={loadMoreCommits}>Load More</Button>
      )}
    </div>
  );
}
