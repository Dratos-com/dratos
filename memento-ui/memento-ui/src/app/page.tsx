import React from 'react';
import MementoFractals from "@/components/memento-fractals-3d";
import { BentoConversationUi } from "@/components/bento-conversation-ui";
import { EnhancedTimePortalConversations } from "@/components/enhanced-time-portal-conversations";
import { SideBySideChatViewer } from "@/components/side-by-side-chat-viewer";
import { TimePortalConversations } from "@/components/time-portal-conversations";
import { mockConversation } from './mockData'; // Add this import

export default function Home() {
  return (
    <div>
      <h1>Memento UI</h1>
      <TimePortalConversations conversationId="your-conversation-id" conversation={mockConversation} currentCommitIndex={0} />
      <MementoFractals conversationHistory={{
        id: 'root',
        content: '',
        children: [
          {
            id: 'unique-id',
            content: 'Conversation content',
            children: [],
            position: [0, 0, 0],
            color: 'defaultColor'
          }
        ],
        position: [0, 0, 0],
        color: ''
      }} />
      <BentoConversationUi />
      <EnhancedTimePortalConversations />
      <SideBySideChatViewer />
    </div>
  );
}
