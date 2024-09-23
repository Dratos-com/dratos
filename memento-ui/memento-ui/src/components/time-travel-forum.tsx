'use client'

import { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Slider } from "@/components/ui/slider";
import { ArrowUpCircle, ArrowDownCircle, MessageSquare, GitBranch, Clock, Zap, TrendingUp, Award } from "lucide-react";
import { useRouter } from 'next/navigation';
import { fetchPosts, votePost } from '@/lib/api';

export function TimeTravelForum() {
  const [timeSlider, setTimeSlider] = useState([100]);
  const [posts, setPosts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const router = useRouter();

  const loadPosts = async () => {
    try {
      setIsLoading(true);
      console.log("Fetching posts from API");
      const fetchedPosts = await fetchPosts();
      console.log("Fetched posts:", fetchedPosts);
      setPosts(fetchedPosts);
    } catch (err) {
      console.error('Error fetching posts:', err);
      setError('Failed to load posts. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadPosts();
  }, []);

  // Add this effect to refresh posts when the component receives focus
  useEffect(() => {
    const handleFocus = () => {
      loadPosts();
    };

    window.addEventListener('focus', handleFocus);
    return () => {
      window.removeEventListener('focus', handleFocus);
    };
  }, []);

  const handleVote = async (id: string, type: 'up' | 'down') => {
    try {
      const updatedPost = await votePost(id, type);
      setPosts(posts.map(post => post.id === id ? updatedPost : post));
    } catch (err) {
      console.error('Error voting on post:', err);
      // Optionally, show an error message to the user
    }
  };

  const handleNewPost = () => {
    router.push('/git-versioned-forum/new-post');
  };

  const renderContent = () => {
    if (isLoading) {
      return <div>Loading posts...</div>;
    }

    if (error) {
      return <div>Error: {error}</div>;
    }

    if (posts.length === 0) {
      return <div>No posts found. Be the first to create a post!</div>;
    }

    return (
      <ScrollArea className="h-[calc(100vh-300px)]">
        {posts.map(post => (
          <Card key={post.id} className="mb-4">
            <CardHeader>
              <CardTitle>{post.title}</CardTitle>
              <CardDescription>Posted by {post.author} • {new Date(post.timestamp).toLocaleString()}</CardDescription>
            </CardHeader>
            <CardContent>
              <p>{post.content}</p>
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="flex items-center space-x-4">
                <Button variant="ghost" size="sm" onClick={() => handleVote(post.id, 'up')}>
                  <ArrowUpCircle className="w-4 h-4 mr-2" />
                  {post.upvotes}
                </Button>
                <Button variant="ghost" size="sm" onClick={() => handleVote(post.id, 'down')}>
                  <ArrowDownCircle className="w-4 h-4 mr-2" />
                  {post.downvotes}
                </Button>
                <Button variant="ghost" size="sm">
                  <MessageSquare className="w-4 h-4 mr-2" />
                  {post.comments} Comments
                </Button>
                <Button variant="ghost" size="sm">
                  <GitBranch className="w-4 h-4 mr-2" />
                  {post.branches} Branches
                </Button>
              </div>
              <Button onClick={() => router.push(`/git-versioned-forum/post/${post.id}`)} variant="outline" size="sm">View Timeline</Button>
            </CardFooter>
          </Card>
        ))}
      </ScrollArea>
    );
  };

  return (
    <div className="container mx-auto p-4">
      <header className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Time Travel Discussion Forum</h1>
        <p className="text-xl text-muted-foreground">Explore the past, present, and future of conversations</p>
      </header>

      <div className="mb-6 flex items-center space-x-4">
        <Input placeholder="Search discussions..." className="flex-grow" />
        <Button onClick={handleNewPost}>New Post</Button>
      </div>

      <div className="mb-6 flex items-center space-x-4">
        <Clock className="w-6 h-6" />
        <span className="font-semibold">Time Slider:</span>
        <Slider
          value={timeSlider}
          onValueChange={setTimeSlider}
          max={100}
          step={1}
          className="w-64"
        />
        <span>{timeSlider}% (Current)</span>
      </div>

      <Tabs defaultValue="hot" className="mb-6">
        <TabsList>
          <TabsTrigger value="hot"><Zap className="w-4 h-4 mr-2" />Hot</TabsTrigger>
          <TabsTrigger value="trending"><TrendingUp className="w-4 h-4 mr-2" />Trending</TabsTrigger>
          <TabsTrigger value="top"><Award className="w-4 h-4 mr-2" />Top</TabsTrigger>
          <TabsTrigger value="new"><Clock className="w-4 h-4 mr-2" />New</TabsTrigger>
        </TabsList>
      </Tabs>

      {renderContent()}
    </div>
  );
}