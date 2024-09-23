import { useRouter } from 'next/router';
import { useEffect, useState } from 'react';
import { PostPageWithMinimap } from "@/components/post-page-with-minimap";
import { fetchPostData } from '@/lib/api';

export default function PostPage() {
  const router = useRouter();
  const { id } = router.query;
  const [post, setPost] = useState(null);

  useEffect(() => {
    if (id) {
      fetchPostData(id).then(setPost);
    }
  }, [id]);

  if (!post) {
    return <div>Loading...</div>;
  }

  return <PostPageWithMinimap post={post} />;
}
