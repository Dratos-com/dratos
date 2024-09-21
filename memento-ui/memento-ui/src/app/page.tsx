import Image from "next/image";
import MementoFractals from "@/components/memento-fractals-3d";
import { BentoConversationUi } from "@/components/bento-conversation-ui";
import { EnhancedTimePortalConversations } from "@/components/enhanced-time-portal-conversations";
import { SideBySideChatViewer } from "@/components/side-by-side-chat-viewer";
import { TimePortalConversations } from "@/components/time-portal-conversations";

export default function Home() {
  return (
    <div>
      <h1>Memento UI</h1>
      <MementoFractals />
      <BentoConversationUi />
      <EnhancedTimePortalConversations />
      <SideBySideChatViewer />
      <TimePortalConversations />
    </div>
  );
}
