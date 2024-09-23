'use client'

import { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { ArrowUpCircle, ArrowDownCircle, MessageSquare, GitBranch, Clock, Reply, Share2 } from "lucide-react";
import { votePost, addComment, createBranch, fetchPostHistory } from '@/lib/api';
import { GitMinimap } from '@/components/git-minimap'; // Assume this component is created separately

export function PostPageWithMinimap({ initialPost }) {
  const [currentPost, setCurrentPost] = useState(initialPost);
  const [timeSlider, setTimeSlider] = useState([100]);
  const [newComment, setNewComment] = useState('');
  const [postHistory, setPostHistory] = useState([]);

  useEffect(() => {
    fetchPostHistory(initialPost.id).then(setPostHistory);
  }, [initialPost.id]);

  const handleVote = async (type: 'up' | 'down') => {
    const updatedPost = await votePost(currentPost.id, type);
    setCurrentPost(updatedPost);
  };

  const handleCommentSubmit = async () => {
    if (newComment.trim() === '') return;
    const updatedPost = await addComment(currentPost.id, newComment);
    setCurrentPost(updatedPost);
    setNewComment('');
  };

  const handleBranch = async (parentId: string) => {
    const newBranch = await createBranch(currentPost.id, parentId);
    console.log(`Branch created: ${newBranch.id}`);
    // Refresh post history after creating a new branch
    fetchPostHistory(initialPost.id).then(setPostHistory);
  };

  const handleTimeTravel = (value: number[]) => {
    setTimeSlider(value);
    const historyIndex = Math.floor((value[0] / 100) * (postHistory.length - 1));
    setCurrentPost(postHistory[historyIndex]);
  };

  return (
    <TooltipProvider>
      <div className="container mx-auto p-4">
        <div className="grid grid-cols-4 gap-4">
          <Card className="col-span-3">
            <CardHeader>
              <CardTitle>{currentPost.title}</CardTitle>
              <CardDescription>
                Posted by {currentPost.author} • {new Date(currentPost.timestamp).toLocaleString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="mb-4">{currentPost.content}</p>
              <div className="flex items-center space-x-4 mb-4">
                <Button variant="ghost" size="sm" onClick={() => handleVote('up')}>
                  <ArrowUpCircle className="w-4 h-4 mr-2" />
                  {currentPost.upvotes}
                </Button>
                <Button variant="ghost" size="sm" onClick={() => handleVote('down')}>
                  <ArrowDownCircle className="w-4 h-4 mr-2" />
                  {currentPost.downvotes}
                </Button>
                <Button variant="ghost" size="sm">
                  <MessageSquare className="w-4 h-4 mr-2" />
                  {currentPost.comments.length} Comments
                </Button>
                <Button variant="ghost" size="sm">
                  <GitBranch className="w-4 h-4 mr-2" />
                  {currentPost.branches.length} Branches
                </Button>
                <Button variant="ghost" size="sm">
                  <Share2 className="w-4 h-4 mr-2" />
                  Share
                </Button>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="sm" onClick={() => handleBranch(currentPost.id)}>
                      <GitBranch className="w-4 h-4 mr-2" />
                      Branch
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Create a new branch from this post</p>
                  </TooltipContent>
                </Tooltip>
              </div>
              <div className="flex items-center space-x-4">
                <Clock className="w-6 h-6" />
                <span className="font-semibold">Time Travel:</span>
                <Slider
                  value={timeSlider}
                  onValueChange={handleTimeTravel}
                  max={100}
                  step={1}
                  className="w-64"
                />
                <span>{timeSlider}% (Current)</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Git Minimap</CardTitle>
            </CardHeader>
            <CardContent>
              <GitMinimap post={currentPost} history={postHistory} />
            </CardContent>
          </Card>
        </div>

        {/* Comments and Branches sections remain the same */}
      </div>
    </TooltipProvider>
  );
}